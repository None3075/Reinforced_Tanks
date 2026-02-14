from src.feature_extractor.OpmizedModel import MultiHeadDiscretePolicy
from stable_baselines3 import PPO

class Agent(PPO):
    def __init__(self, env, args: dict, policy_kwargs=None):
        super().__init__(
            policy=MultiHeadDiscretePolicy, 
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate = args["learning_rate"],
            n_steps = args["n_steps"],
            batch_size = args["batch_size"],
            n_epochs = args["n_epochs"],
            gamma = args["gamma"],
            gae_lambda = args["gae_lambda"],
            clip_range = args["clip_range"],
            clip_range_vf = args["clip_range_vf"],
            normalize_advantage = args["normalize_advantage"],
            ent_coef = args["ent_coef"],
            vf_coef = args["vf_coef"],
            max_grad_norm = args["max_grad_norm"],
            use_sde = args["use_sde"],
            sde_sample_freq = args["sde_sample_freq"],
            stats_window_size = args["stats_window_size"],
            verbose = args["verbose"],
            seed = args["seed"],
            target_kl = args["target_kl"],
            tensorboard_log="./tensorboard/"
        )

    def predict(self, observation: dict, deterministic: bool):
        action, _states = super().predict(observation, deterministic=deterministic)
        return action, _states
