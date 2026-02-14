import numpy as np
from typing import Literal
import math

class OBSParser:
    
    @staticmethod
    def parse_tanks(perception: dict):
            tanks = []
            bullets = []
            for tank in perception:
                # Process tank information
                info_vector = []
                info_vector.append(int(tank["alive"]))
                info_vector.append(tank["angle"])
                info_vector.append(tank["x"])
                info_vector.append(tank["y"])
                tanks.append(info_vector)

                info_vector = []
                info_vector.append(int(tank["bullets"]["alive"]))
                info_vector.append(tank["bullets"]["angle"])
                info_vector.append(tank["bullets"]["bounces"])
                info_vector.append(tank["bullets"]["x"])
                info_vector.append(tank["bullets"]["y"])
                bullets.append(info_vector)
            return np.array(tanks), np.array(bullets)
    
    @staticmethod
    def parse_team(perception: dict):
        own_tanks, own_bullets = OBSParser.parse_tanks(perception["perceiving_state"]["tanks"])
        enemy_tanks, enemy_bullets = OBSParser.parse_tanks(perception["perceiving_state"]["enemy_tanks"])
        return [own_tanks, own_bullets], [enemy_tanks, enemy_bullets]

    @staticmethod
    def get_angle_to_target(source_x, source_y, source_angle, target_x, target_y):
        dx = target_x - source_x
        dy = target_y - source_y
        
        target_angle_rad = math.atan2(dy, dx)
        target_angle_deg = math.degrees(target_angle_rad)
        
        angle_diff = target_angle_deg - source_angle
        
        angle_diff = (angle_diff + 360) % 360
        
        return angle_diff

    @staticmethod
    def to_relative_observation(obs: dict) -> dict:
        # Get first own tank as reference
        ref_tank = obs["own_tanks"][0] if isinstance(obs["own_tanks"], np.ndarray) else obs["own_tanks"][0]
        ref_alive, ref_angle, ref_x, ref_y = ref_tank[0], ref_tank[1], ref_tank[2], ref_tank[3]
        
        # If reference tank is dead, return original observation
        if ref_alive == 0:
            return obs
        
        relative_obs = {}
        
        # own tanks
        own_tanks = obs["own_tanks"].copy() if isinstance(obs["own_tanks"], np.ndarray) else np.array([t.copy() for t in obs["own_tanks"]])
        for i in range(len(own_tanks)):
            if own_tanks[i][0] == 1:
                if i == 0:
                    own_tanks[i][1] = 0
                    own_tanks[i][2] = 0
                    own_tanks[i][3] = 0
                else:
                    own_tanks[i][1] = OBSParser.get_angle_to_target(ref_x, ref_y, ref_angle, own_tanks[i][2], own_tanks[i][3])
                    own_tanks[i][2] = own_tanks[i][2] - ref_x
                    own_tanks[i][3] = own_tanks[i][3] - ref_y
        relative_obs["own_tanks"] = own_tanks.astype(np.float32)
        
        # own bullets
        own_bullets = obs["own_bullets"].copy() if isinstance(obs["own_bullets"], np.ndarray) else np.array([b.copy() for b in obs["own_bullets"]])
        for i in range(len(own_bullets)):
            if own_bullets[i][0] == 1:
                own_bullets[i][1] = OBSParser.get_angle_to_target(ref_x, ref_y, ref_angle, own_bullets[i][3], own_bullets[i][4])
                own_bullets[i][3] = own_bullets[i][3] - ref_x
                own_bullets[i][4] = own_bullets[i][4] - ref_y
        relative_obs["own_bullets"] = own_bullets.astype(np.float32)
        
        # enemy tanks
        enemy_tanks = obs["enemy_tanks"].copy() if isinstance(obs["enemy_tanks"], np.ndarray) else np.array([t.copy() for t in obs["enemy_tanks"]])
        for i in range(len(enemy_tanks)):
            if enemy_tanks[i][0] == 1:
                enemy_tanks[i][1] = OBSParser.get_angle_to_target(ref_x, ref_y, ref_angle, enemy_tanks[i][2], enemy_tanks[i][3])
                enemy_tanks[i][2] = enemy_tanks[i][2] - ref_x
                enemy_tanks[i][3] = enemy_tanks[i][3] - ref_y
        relative_obs["enemy_tanks"] = enemy_tanks.astype(np.float32)
        
        # enemy bullets
        enemy_bullets = obs["enemy_bullets"].copy() if isinstance(obs["enemy_bullets"], np.ndarray) else np.array([b.copy() for b in obs["enemy_bullets"]])
        for i in range(len(enemy_bullets)):
            if enemy_bullets[i][0] == 1:
                enemy_bullets[i][1] = OBSParser.get_angle_to_target(ref_x, ref_y, ref_angle, enemy_bullets[i][3], enemy_bullets[i][4])
                enemy_bullets[i][3] = enemy_bullets[i][3] - ref_x
                enemy_bullets[i][4] = enemy_bullets[i][4] - ref_y
        relative_obs["enemy_bullets"] = enemy_bullets.astype(np.float32)
        
        return relative_obs

    @staticmethod
    def get_obs(state, team: Literal["red", "blue"] = "red")->np.array:
        (own_tanks, own_bullets), (enemy_tanks, enemy_bullets) = OBSParser.parse_team(state[f"{team}_team"])
        
        obs = {
            "own_tanks": own_tanks.astype(np.float32),
            "own_bullets": own_bullets.astype(np.float32),
            "enemy_tanks": enemy_tanks.astype(np.float32),
            "enemy_bullets": enemy_bullets.astype(np.float32),
        }
        return obs

    @staticmethod
    def parse_obs(obs: dict) -> dict:

        ret = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                ret[key] = value.tolist()
            else:
                ret[key] = value
            for i, v in enumerate(ret[key]):
                if isinstance(v, np.ndarray):
                    ret[key][i] = v.tolist()
                if isinstance(value, np.ndarray):
                    value = value.tolist()
        return ret
    
if __name__ == "__main__":
    obs = {'enemy_bullets': np.array([
        [  0., 180.,   0.,
            498.75995,
            114.973465
        ],
        [  0., 180.,   0.,
            507.6125,
            234.0099
        ]
    ],
      dtype=np.float32),
  'enemy_tanks': np.array([
        [  1., 180.,
            498.75995,
            114.973465
        ],
        [  1., 180.,
            507.6125,
            234.0099
        ]
    ], dtype=np.float32),
  'own_bullets': [np.array([  0.,   0.,   0.,
            403.16388,
            126.2153
        ],
      dtype=np.float32),
                  np.array([  0.,   0.,   0.,
            420.32227,
            417.03937
        ],
      dtype=np.float32)
    ],
  'own_tanks': [np.array([  1.,   0.,
            403.16388,
            126.2153
        ], dtype=np.float32),
                np.array([  1.,   0.,
            420.32227,
            417.03937
        ], dtype=np.float32)
    ]
}
    relative_obs = OBSParser.to_relative_observation(obs)
    print("Original Observation:")
    print(obs)
    print("\nRelative Observation:")
    print(relative_obs)