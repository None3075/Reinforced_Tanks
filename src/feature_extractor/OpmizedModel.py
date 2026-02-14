import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Sequence, Type, Literal
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution

def normalized_pos(pos :torch.tensor):
    x_max, y_max = 948.75, 528.5491071428571
    x = pos[:, :, -2:-1]
    y = pos[:, :, -1:]
    x = (x) / x_max
    y = (y) / y_max
    return torch.cat([x, y], dim=2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def describe_tensor(features: torch.Tensor, name: str = "features"):
    """Safely print basic stats for a tensor, guarding against empty/NaN/Inf inputs."""
    if features is None:
        print(f"{name}: None")
        return
    # Handle empty tensors
    if features.numel() == 0:
        print(f"{name}: empty tensor with shape={tuple(features.shape)}")
        return
    # Detach, ensure float for stats
    f = features.detach()
    try:
        f = f.to(dtype=torch.float32)
    except Exception:
        pass
    # Filter to finite values only
    finite_mask = torch.isfinite(f)
    if not finite_mask.any():
        print(f"{name}: no finite values (shape={tuple(f.shape)})")
        return
    f = f[finite_mask]
    print(
        f"{name}: max={f.max().item():.3f}, min={f.min().item():.3f}, "
        f"mean={f.mean().item():.3f}, std={f.std().item():.3f}, shape={tuple(features.shape)}"
    )

class T_B(nn.Module):
    tank_speed: int = 7
    tank_radius: int = 22

    bullet_speed: int = 4

    t = (tank_radius-tank_speed) + bullet_speed #Unused
    embedding_dim: int
    def __init__(
        self,
        embedding_dim: int,
        proyector_in: int = 10,                 # dimensión de entrada al proyector
        proyector_hidden: Sequence[int] = (512, 256, 128, 64),  # lista de capas ocultas
        activation: Type[nn.Module] = nn.ReLU, # activación entre capas
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proyector = self._make_mlp(
            [proyector_in, *proyector_hidden, self.embedding_dim],
            activation=activation
        )
        self._init_xavier()

    @staticmethod
    def _make_mlp(sizes: Sequence[int], activation: Type[nn.Module] = nn.ReLU) -> nn.Sequential:
        """Construye un MLP Linear-Act-...-Linear según sizes=[in, h1, ..., out]."""
        layers = []
        for i in range(len(sizes) - 1):
            in_f, out_f = sizes[i], sizes[i + 1]
            layers.append(nn.Linear(in_f, out_f, bias=True))
            # Añade activación salvo después de la última Linear
            if i < len(sizes) - 2:
                layers.append(activation())
        return nn.Sequential(*layers)

    def _init_xavier(self):
        for m in self.proyector:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # o nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def ensure_batch(self, main_tank: torch.tensor, bullet: torch.tensor):
        if main_tank.dim() == 1:
            main_tank = main_tank.unsqueeze(0)
        if bullet.dim() == 1:
            bullet = bullet.unsqueeze(0)
        return main_tank, bullet

    def ensure_angles(self, main_tank: torch.tensor, bullet: torch.tensor):
        main_tank_angle = main_tank[:, -3:-2]   # (B,1)
        bullet_angle    = bullet[:, -4:-3]
        mask_main   = (main_tank_angle == 90) | (main_tank_angle == 270)
        mask_bullet = (bullet_angle    == 90) | (bullet_angle    == 270)

        eps = 1e-4
        # Ajuste sin in-place duro (mejor para autograd)
        main_tank_angle = torch.where(mask_main,   main_tank_angle - eps, main_tank_angle)
        bullet_angle    = torch.where(mask_bullet, bullet_angle    - eps, bullet_angle)

        return main_tank_angle, bullet_angle
    
    def calculate_collider_point(self, main_tank_pos: torch.tensor, bullet_pos: torch.tensor, main_tank_angle: torch.tensor, bullet_angle: torch.tensor):

        m_tank = torch.tan(torch.deg2rad(main_tank_angle)).flatten(start_dim=0)
        m_bullets = torch.tan(torch.deg2rad(bullet_angle)).flatten(start_dim=0)

        n_tank = main_tank_pos[:, -1] - (m_tank * main_tank_pos[:, 0])
        n_bullets = bullet_pos[:, -1] - (m_bullets * bullet_pos[:, 0])

        Ix = (n_tank - n_bullets) / (m_bullets - m_tank)
        Iy = ((n_tank * m_bullets) - (n_bullets * m_tank)) / (m_bullets - m_tank)

        trajectory_dist = (torch.abs(m_bullets*main_tank_pos[:, 0] - main_tank_pos[:, -1] + n_bullets)) / torch.sqrt(m_bullets**2 + 1)
        I = torch.stack([Ix, Iy], dim=1)

        return I, trajectory_dist

    def calculate_collision_time(self, main_tank_pos, bullet_pos, main_tank_angle, bullet_angle):
        collision_point, trajectory_dist = self.calculate_collider_point(main_tank_pos=main_tank_pos, 
                                                        main_tank_angle=main_tank_angle, 
                                                        bullet_pos=bullet_pos, 
                                                        bullet_angle=bullet_angle)
        tank_dist = torch.linalg.norm(main_tank_pos - collision_point, dim=1)
        tank_collision_time = tank_dist / self.tank_speed
        tank_collision_time = self.ensure_collision_time(tank_collision_time)

        bullet_dist = torch.linalg.norm(bullet_pos - collision_point, dim=1)
        bullet_collision_time = bullet_dist / self.bullet_speed
        bullet_collision_time = self.ensure_collision_time(bullet_collision_time)

        dist = torch.linalg.norm(bullet_pos - main_tank_pos, dim=1)
        return tank_collision_time, bullet_collision_time, dist, trajectory_dist

    def ensure_collision_time(self, collision_time):
        bad = ~torch.isfinite(collision_time)
        collision_time = torch.where(
            bad,
            torch.full_like(collision_time, -1),
            collision_time
        )
        return collision_time
    
    def forward(self, main_tank: torch.tensor, bullet: torch.tensor):

        main_tank, bullet = self.ensure_batch(main_tank=main_tank, bullet=bullet)
        main_tank_pos = main_tank[:, -2:]
        bullet_pos = bullet[:, -2:]
        main_tank_angle, bullet_angle = self.ensure_angles(main_tank=main_tank, bullet=bullet)
        collt_t, collt_b, dist, trajectory_dist = self.calculate_collision_time(main_tank_pos, bullet_pos, main_tank_angle, bullet_angle)

        in_features_a = torch.stack([
            #collt_t, collt_b, 
            dist, trajectory_dist], dim=1)
        in_features_b = torch.cat([main_tank_pos, bullet_pos], dim=1)
        main_tank_angle_rad = torch.deg2rad(main_tank_angle)
        bullet_angle_rad = torch.deg2rad(bullet_angle)
        in_features_c = torch.cat([torch.cos(main_tank_angle_rad), torch.cos(bullet_angle_rad), torch.sin(main_tank_angle_rad), torch.sin(bullet_angle_rad)], dim=1)
        in_features = torch.cat([in_features_a, in_features_b, in_features_c], dim=1)
        ret = self.proyector(in_features)
        return ret

class T_T(nn.Module):
    tank_speed: int = 7
    tank_radius: int = 22

    bullet_speed: int = 4

    center = normalized_pos(torch.tensor([[[480.0, 270.0]]])).to(DEVICE)[0]
    def __init__(
        self,
        embedding_dim: int,
        proyector_in: int = 11,                 # dimensión de entrada al proyector
        proyector_hidden: Sequence[int] = (512, 256, 128, 64),  # lista de capas ocultas
        activation: Type[nn.Module] = nn.ReLU, # activación entre capas
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proyector = self._make_mlp(
            [proyector_in, *proyector_hidden, self.embedding_dim],
            activation=activation
        )

    @staticmethod
    def _make_mlp(sizes: Sequence[int], activation: Type[nn.Module] = nn.ReLU) -> nn.Sequential:
        """Construye un MLP Linear-Act-...-Linear según sizes=[in, h1, ..., out]."""
        layers = []
        for i in range(len(sizes) - 1):
            in_f, out_f = sizes[i], sizes[i + 1]
            layers.append(nn.Linear(in_f, out_f, bias=True))
            # Añade activación salvo después de la última Linear
            if i < len(sizes) - 2:
                layers.append(activation())
        proj = nn.Sequential(*layers)
        for x in proj:
            if isinstance(x, nn.Linear):
                nn.init.xavier_uniform_(x.weight)   # o nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(x.bias)
        return proj

    def angle_to_vec2(self, theta, r=1.0, degrees=True):
        th = torch.as_tensor(theta, dtype=torch.float32)
        if degrees:
            th = torch.deg2rad(th)
        x = r * torch.cos(th)
        y = r * torch.sin(th)
        return torch.stack((x, y), dim=-1) 

    def ensure_batch(self, main_tank: torch.tensor, tank: torch.tensor):
        if main_tank.dim() == 1:
            main_tank = main_tank.unsqueeze(0)
        if tank.dim() == 1:
            tank = tank.unsqueeze(0)
        return main_tank, tank

    def ensure_angles(self, main_tank: torch.tensor, tank: torch.tensor):
        main_tank_angle = main_tank[:, -3:-2]   # (B,1)
        tank_angle    = tank[:, -3:-2]
        mask_main   = (main_tank_angle == 90) | (main_tank_angle == 270)
        mask_tank = (tank    == 90) | (tank    == 270)

        eps = 1e-4
        # Ajuste sin in-place duro (mejor para autograd)
        main_tank_angle = torch.where(mask_main,   main_tank_angle - eps, main_tank_angle)
        tank_angle    = torch.where(mask_tank, tank_angle    - eps, tank_angle)

        return main_tank_angle, tank_angle   
    
    def forward(self, main_tank: torch.tensor, tank: torch.tensor):
        main_tank_pos = main_tank[:, -2:]
        tank_pos = tank[:, -2:]
        distance = torch.linalg.norm(main_tank_pos - tank_pos, dim=1, keepdim=True)
        distance_to_center_main = torch.linalg.norm(main_tank_pos - self.center, dim=1, keepdim=True)
        distance_to_center_tank = torch.linalg.norm(tank_pos - self.center, dim=1, keepdim=True)
        distance_to_center_diff = distance_to_center_main - distance_to_center_tank
        main_tank_angle = main_tank[:, 1:2]
        tank_angle      = tank[:, 1:2]
        diff_angle = torch.deg2rad(torch.abs(main_tank_angle - tank_angle + 180) % 360 - 180)

        in_features_b = torch.cat([main_tank_pos, tank_pos, distance, distance_to_center_diff], dim=1)

        main_tank_angle_rad = torch.deg2rad(main_tank_angle)
        tank_angle_rad = torch.deg2rad(tank_angle)
        in_features_c = torch.cat([torch.cos(main_tank_angle_rad), torch.cos(tank_angle_rad), torch.sin(main_tank_angle_rad), torch.sin(tank_angle_rad)], dim=1)
        
        in_features = torch.cat([in_features_b, in_features_c, diff_angle], dim=1)
        return self.proyector(in_features)

class OptimizedModel(BaseFeaturesExtractor):

    main_with_team_tank: T_T
    main_with_enem_tank: T_T
    main_with_bullets: T_B
    bullet_proj: nn.Sequential
    team_proj: nn.Sequential
    enemy_proj: nn.Sequential
    final_proj: nn.Sequential

    bullets_embedding_dim: int
    tank_embedding_dim: int

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, 
                 observation_space, features_dim,
                 T_B_hidden_layers: Sequence[int] = (16, 16, 8),
                 T_B_embedding_dim: int = 6,
                 T_T_hidden_layers: Sequence[int] = (16, 16, 8),
                 T_T_embedding_dim: int = 6,
                 T_B_activation:  Type[nn.Module] = nn.ReLU,
                 T_T_activation:  Type[nn.Module] = nn.ReLU
                ):
        super(OptimizedModel, self).__init__(observation_space, features_dim)
        self.main_with_team_tank =  T_T(embedding_dim=T_T_embedding_dim, proyector_hidden=T_T_hidden_layers) 
        self.main_with_enem_tank =  T_T(embedding_dim=T_T_embedding_dim, proyector_hidden=T_T_hidden_layers) 
        self.main_with_bullets = T_B(embedding_dim=T_B_embedding_dim, proyector_hidden=T_B_hidden_layers) 
        self.bullets_embedding_dim = T_B_embedding_dim
        self.tank_embedding_dim = T_B_embedding_dim

        self.bullet_proj = nn.Sequential(
            nn.Linear(4*self.bullets_embedding_dim, 2*self.bullets_embedding_dim),
            nn.Linear(2*self.bullets_embedding_dim, self.bullets_embedding_dim),
        ) 
        self.team_proj = nn.Sequential(
            nn.Identity()
        ) 
        self.enemy_proj = nn.Sequential(
            nn.Linear(2*self.tank_embedding_dim, 2*self.tank_embedding_dim),
            nn.Linear(2*self.tank_embedding_dim, self.tank_embedding_dim),
        ) 
        self.final_proj = nn.Sequential(
            nn.Linear(self.tank_embedding_dim * 2 + self.bullets_embedding_dim, features_dim),
        )

        #xavier init
        for m in self.bullet_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # o nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.enemy_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # o nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)  
        for m in self.final_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # o nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.team_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # o nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def batched_bullets_processing(self, obs):
        B, _, _ = torch.as_tensor(obs["own_tanks"]).shape
        maintank_tensor = obs["own_tanks"][:, 0, :]
        main_tank_input_tensor = maintank_tensor.repeat_interleave(4, dim=0)
        input_bullets = torch.zeros([B*4, 5])

        for batch_id in range(B):
            input_bullets[4*batch_id:4*batch_id+2, :] = obs["own_bullets"][batch_id]
            input_bullets[4*batch_id+2:4*batch_id+4, :] = obs["enemy_bullets"][batch_id]

        mask = input_bullets[:, 0].to(torch.bool)
        result = torch.zeros([B*4, self.bullets_embedding_dim]).to(DEVICE)
        result[mask] = self.main_with_bullets.forward(main_tank=main_tank_input_tensor[mask].to(DEVICE), bullet=input_bullets[mask].to(DEVICE))
        result = result.view(-1, 4, self.bullets_embedding_dim).flatten(1)
        result = self.bullet_proj(result)
        return result
    
    def batched_team_tank_processing(self, obs):
        B, _, _ = torch.as_tensor(obs["own_tanks"]).shape
        maintank_tensor = obs["own_tanks"][:, 0, :]
        main_tank_input_tensor = maintank_tensor.repeat_interleave(1, dim=0)
        input_tank = torch.zeros([B, 4])

        for batch_id in range(B):
            tmp = obs[f"own_tanks"][batch_id].clone()[-1:]
            input_tank[batch_id, :] = tmp
        mask = input_tank[:, 0].to(torch.bool)
        result = torch.zeros([B, self.tank_embedding_dim]).to(DEVICE)

        result[mask] = self.main_with_team_tank.forward(main_tank=main_tank_input_tensor[mask].to(DEVICE), tank=input_tank[mask].to(DEVICE))
        result = self.team_proj(result)
        return result
    
    def batched_enemy_tank_processing(self, obs):
        B, _, _ = torch.as_tensor(obs["own_tanks"]).shape
        maintank_tensor = obs["own_tanks"][:, 0, :]
        main_tank_input_tensor = maintank_tensor.repeat_interleave(2, dim=0)
        input_tank = torch.zeros([B*2, 4])

        for batch_id in range(B):
            tmp = obs[f"enemy_tanks"][batch_id].clone()
            input_tank[2*batch_id:2*batch_id+2, :] = tmp

        mask = input_tank[:, 0].to(torch.bool)
        result = torch.zeros([B*2, self.tank_embedding_dim]).to(DEVICE)

        result[mask] = self.main_with_enem_tank.forward(main_tank=main_tank_input_tensor[mask].to(DEVICE), tank=input_tank[mask].to(DEVICE))

        result = result.view(-1,2,self.tank_embedding_dim).flatten(1)
        result = self.enemy_proj(result)
        return result

    def forward(self, obs):
        obs["own_tanks"][:, :, -2:] = normalized_pos(obs["own_tanks"][:, :, -2:])
        obs["enemy_tanks"][:, :, -2:] = normalized_pos(obs["enemy_tanks"][:, :, -2:])
        obs["own_bullets"][:, :, -2:] = normalized_pos(obs["own_bullets"][:, :, -2:])
        obs["enemy_bullets"][:, :, -2:] = normalized_pos(obs["enemy_bullets"][:, :, -2:])

        bullets_embeddins = self.batched_bullets_processing(obs=obs)
        team_tank_embeddins = self.batched_team_tank_processing(obs=obs)
        enemy_tank_embeddins = self.batched_enemy_tank_processing(obs=obs)

        input_embedding = torch.cat([bullets_embeddins, team_tank_embeddins, enemy_tank_embeddins], dim=1)
        ret = self.final_proj(input_embedding)
        return ret
    
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MultiHeadDiscretePolicy(MultiInputActorCriticPolicy):
    """
    Custom policy for MultiDiscrete action space [3, 3, 2].
    Can use OptimizedModel or default feature extractor, then separate MLP heads for each action dimension.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Only set default features_extractor_kwargs if not provided
        # Do NOT force OptimizedModel - let user decide or use SB3's default (CombinedExtractor for Dict spaces)
        # if 'features_extractor_kwargs' not in kwargs:
        #     # Set default features_dim only if using a custom extractor that needs it
        #     if 'features_extractor_class' in kwargs:
        #         kwargs['features_extractor_kwargs'] = {'features_dim': 128}
        
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # Action space should be MultiDiscrete([3, 3, 2])
        self.action_dims = action_space.nvec.tolist()  # [3, 3, 2]
        
        # Get the features dimension from the extractor
        features_dim = self.features_extractor.features_dim
        
        # Build separate action heads for each tank (3 actions total)
        # Each head is an MLP that outputs logits for its action dimension
        net_arch_pi = self.net_arch.get('pi', [64, 64]) if isinstance(self.net_arch, dict) else [64, 64]
        
        # Action head for tank 0 (3 discrete actions)
        self.action_head_0 = self._build_action_head(features_dim, net_arch_pi, self.action_dims[0])
        
        # Action head for tank 1 (3 discrete actions)
        self.action_head_1 = self._build_action_head(features_dim, net_arch_pi, self.action_dims[1])
        
        # Action head for tank 2 (2 discrete actions)
        self.action_head_2 = self._build_action_head(features_dim, net_arch_pi, self.action_dims[2])
        
        # Value head (critic)
        net_arch_vf = self.net_arch.get('vf', [64, 64]) if isinstance(self.net_arch, dict) else [64, 64]
        self.value_head = self._build_value_head(features_dim, net_arch_vf)
        
        # Action distribution
        self.action_dist = MultiCategoricalDistribution(self.action_dims)
    
    def _build_action_head(self, input_dim: int, hidden_layers: Sequence[int], output_dim: int) -> nn.Sequential:
        """Build an MLP head for a single action dimension with Xavier initialization."""
        layers = []
        last_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            last_dim = hidden_dim
        
        # Final layer outputs logits for this action dimension
        layers.append(nn.Linear(last_dim, output_dim))
        head = nn.Sequential(*layers)
        # Xavier initialization for all Linear layers
        for m in head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        return head
    
    def _build_value_head(self, input_dim: int, hidden_layers: Sequence[int]) -> nn.Sequential:
        """Build the value network (critic)."""
        layers = []
        last_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        
        # Final layer outputs a single value
        layers.append(nn.Linear(last_dim, 1))
        
        head =  nn.Sequential(*layers)
        # Xavier initialization for all Linear layers
        for m in head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        return head
    
    def forward(self, obs, deterministic: bool = False):
        # Extract features using OptimizedModel
        features = self.extract_features(obs)

        # Get logits from each action head
        logits_0 = self.action_head_0(features)  # (batch, 3)
        logits_1 = self.action_head_1(features)  # (batch, 3)
        logits_2 = self.shoot_gate(obs, features)

        all_logits = torch.cat([logits_0, logits_1, logits_2], dim=1)
        # Create distribution and sample
        distribution = self.action_dist.proba_distribution(all_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        #Clamp value predictions to prevent explosion
        values = self.value_head(features)
        return actions, values, log_prob

    def shoot_gate(self, obs, features):
        bullets = obs["own_bullets"]
        if bullets.dim() == 2:
            bullets = bullets.unsqueeze(1)

        mask = torch.logical_not(bullets[:, 0, 0].to(torch.bool)) # Mask of available bullets
        logits_2 = torch.zeros((features.shape[0], 2)).to(DEVICE)
        logits_2[mask] = self.action_head_2(features[mask])
        logits_2[torch.logical_not(mask)] = torch.tensor([-1e8, 1e8]).to(DEVICE)  # Force "do not shoot" if no bullets available
        return logits_2
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy.
        Used during training.
        """
        features = self.extract_features(obs)
        
        logits_0 = self.action_head_0(features)  # (batch, 3)
        logits_1 = self.action_head_1(features)  # (batch, 3)
        logits_2 = self.shoot_gate(obs, features) # (batch, 2)

        all_logits = torch.cat([logits_0, logits_1, logits_2], dim=1)
        
        # Create distribution
        distribution = self.action_dist.proba_distribution(all_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Compute value
        values = self.value_head(features)
        
        return values, log_prob, entropy
    
    def _predict(self, observation, deterministic: bool = False):
        features = self.extract_features(observation)
        
        # Get logits from each action head
        logits_0 = self.action_head_0(features)  # (batch, 3)
        logits_1 = self.action_head_1(features)  # (batch, 3)
        logits_2 = self.shoot_gate(observation, features)  # (batch, 2)
        all_logits = torch.cat([logits_0, logits_1, logits_2], dim=1)
        # Create distribution and get actions
        distribution = self.action_dist.proba_distribution(all_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions
    
    def predict_values(self, obs):
        """
        Get the estimated values according to the current policy.
        """
        features = self.extract_features(obs)
        values = self.value_head(features)
        return values


if __name__ == '__main__':
    observations = {
        "own_tanks": torch.Tensor([[
            [1, 90.0, 200.0, 150.0],
            [0, 0.0,   0.0,   0.0]
        ],[
            [1, 90.0, 777.0, 150.0],
            [1, 0.0,   0.0,   0.0]
        ]
        ]), 

        "enemy_tanks": torch.Tensor([[
            [1, 270.0, 800.0, 500.0],
            [1, 135.0, 600.0, 300.0]
        ],[
            [1, 270.0, 800.0, 500.0],
            [1, 135.0, 600.0, 300.0]
        ]]), 

        "own_bullets": torch.Tensor([[
            [1, 90.0, 2, 250.0, 150.0],
            [0, 0.0,  0,   0.0,   0.0]
        ],[
            [1, 90.0, 2, 777.0, 150.0],
            [0, 0.0,  0,   0.0,   0.0]
        ]
        ]),  

        "enemy_bullets": torch.Tensor([[
            [1, 270.0, 3, 850.0, 500.0],
            [0,   0.0, 0,   0.0,   0.0]
        ],[
            [1, 90.0, 2, 250.0, 150.0],
            [0, 0.0,  0,   777.0,   0.0]
        ]
        ])  
    }

    observations_1 = {
        "own_tanks": torch.Tensor([[
            [1, 90.0, 200.0, 150.0],
            [0, 0.0,   0.0,   0.0]
        ]
        ]), 

        "enemy_tanks": torch.Tensor([[
            [1, 270.0, 800.0, 500.0],
            [1, 135.0, 600.0, 300.0]
        ]]), 

        "own_bullets": torch.Tensor([[
            [1, 90.0, 2, 250.0, 150.0],
            [0, 0.0,  0,   0.0,   0.0]
        ]
        ]),  

        "enemy_bullets": torch.Tensor([[
            [1, 270.0, 3, 850.0, 500.0],
            [0,   0.0, 0,   0.0,   0.0]
        ]
        ])  
    }
        
    model = OptimizedModel(observation_space=None, features_dim=8).to("cuda")
    model(observations)
    model(observations_1)
    print(model.n_parameters())

