from typing import Literal, Callable
from stable_baselines3 import PPO
import os
import json
import wandb

from src.ReinforcedTanksEnv import ReinforcedTanksEnv
from src.CustomWandbCallback import CustomWandbCallback

class LearningSchedule:
    
    epochs: list[int]
    reward_functions: list[Callable[[dict, dict, int], int]]
    steps = list[tuple]
    
    def __init__(self, epochs: list[int], reward_functions: list[Callable[[dict, dict, int], int]]):
        assert len(epochs) == len(reward_functions)
        self.epochs = epochs
        self.reward_functions = reward_functions
        self.steps = [(epochs[i], reward_functions[i]) for i in range(len(epochs))]
    
    def __iter__(self):
        for step in self.steps:
            yield step
    
class Trainer:
    finished: bool = False
    model : PPO
    learning_schedule: LearningSchedule
    args: dict
    env: ReinforcedTanksEnv
    
    checkpoint_folder: str = None
    saved_model_folder: str = None
    
    def __init__(self, model: PPO, env: ReinforcedTanksEnv, args):
        self.model = model
        
        self.learning_schedule = LearningSchedule(epochs=args["epochs"],
                                                  reward_functions=args["reward_functions"]
                                                  )
        self.args = args
        self.env = env
        
        if not os.path.exists("example_state.json"):
            with open("example_state.json", "w", encoding="utf-8") as f:
                json.dump(env.parse_obs(env._get_obs()), f, ensure_ascii=False, indent=4)
        
        chroot = os.getcwd()
        
        self.checkpoint_folder = os.path.join(chroot, "checkpoints")
        os.makedirs(
            self.checkpoint_folder,
            exist_ok=True
                    )
        self.saved_model_folder = os.path.join(chroot, "models")
        os.makedirs(
            self.saved_model_folder,
            exist_ok=True
                    )
        
    def train_model(self):
        if self.args["use_wandb"]:
            wandb.login()
            run = wandb.init(
                project=self.args["project_name"], 
                entity=self.args["entity_name"], 
                config=self.args, 
                sync_tensorboard=True,
                monitor_gym=True
            )
            
            wandb_callback = CustomWandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/",
                verbose=2,
                log="all"
            )
        else:
            wandb_callback = None
            
        for learning_step in self.learning_schedule:
            epochs, reward_function = learning_step
            self.env.set_reward_function(reward_function)
            self.model.get_env().env_method("set_reward_function", reward_function)
            for stage in range(3):
                self.model.get_env().env_method("set_stage", stage)
                for epoch in range(epochs):
                    self.model.learn(
                        self.args["timesteps"], 
                        reset_num_timesteps=False,
                        callback=wandb_callback
                    )
                    visualization_flag = (epoch % self.args["visualization_stride"] == 0)
                    save_flag = (epoch % self.args["checkpoint_stride"] == 0)
                    yield visualization_flag, save_flag, stage
        
        self.finished = True
        if self.args["use_wandb"]:
            run.finish()
    
    def save_model(self, epoch: int, stage: int, data: dict) -> str:
        '''Saves the model and data to the checkpoint/ folder and also saves a copy to the enemy_model/ folder'''
        
        save_folder = os.path.join(self.checkpoint_folder, f"stage_{stage}/epoch_{epoch}")
        os.makedirs(
            save_folder,
            exist_ok=True
                    )
        os.makedirs(
            "enemy_model/",
            exist_ok=True
            )
        self.model.save(os.path.join(save_folder, "model.zip"))
        
        self.model.save(f"enemy_model/model_{len(os.listdir('enemy_model/'))}.zip")

        with open(os.path.join(save_folder, "data.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        return os.path.join(save_folder, "model.zip")
                
    def load_model(self, path: str):
        model = PPO.load(
            path=path
        )
        return model

