import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Callable
import os
from stable_baselines3 import PPO

from src.Game import Game
from src.ObservationParser import OBSParser
from src.GameRenderer import GameRenderer
import random

class ReinforcedTanksEnv(gym.Env):
    
    game: Game
    reward_function: Callable[[dict, int], int] = None
    rendering: bool
    time_limit : int 
    random_start : bool
    
    MAP_HEIGHT = 1080 / 2
    MAP_WIDTH = 1920 / 2

    static_counter: int = 2
    curr_obs = None
    blue_action = None
    update_flag = False

    renderer: GameRenderer = None
    
    episode_results = {"red_wins": 0, "blue_wins": 0, "draws": 0, "total": 0}

    def __init__(self, time_limit: int, random_start: bool, rendering: bool = False):
        super(ReinforcedTanksEnv, self).__init__()
        self.game = Game(time_limit=time_limit)
        self.time_limit = time_limit
        self.random_start = random_start
        self.rendering = rendering

        if rendering:
            self.renderer = GameRenderer()
        
        np.random.seed(random.randint(0, 10000))
        print("Environment initialized with seed:", np.random.get_state()[1][0])

        # Tank: alive, angle (degrees), x, y
        tank_low = np.array([0, 0, 0, 0], dtype=np.float32)
        tank_high = np.array([1, 360.0, 948.75, 528.5491071428571], dtype=np.float32)

        # Bullet: alive, angle, bounces, x, y
        bullet_low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        bullet_high = np.array([1, 360.0, 3, 948.75, 528.5491071428571], dtype=np.float32)

        self.observation_space = spaces.Dict({
            "own_tanks": spaces.Box(low=np.tile(tank_low, (2, 1)), high=np.tile(tank_high, (2, 1)), dtype=np.float32),
            "own_bullets": spaces.Box(low=np.tile(bullet_low, (2, 1)), high=np.tile(bullet_high, (2, 1)), dtype=np.float32),
            "enemy_tanks": spaces.Box(low=np.tile(tank_low, (2, 1)), high=np.tile(tank_high, (2, 1)), dtype=np.float32),
            "enemy_bullets": spaces.Box(low=np.tile(bullet_low, (2, 1)), high=np.tile(bullet_high, (2, 1)), dtype=np.float32),
        })
        
        # 1 tanks × (move + turn + shoot) = 3 discrete actions
        # Move:    3 options → 0 = still, 1 = forward,  2 = backward
        # Turn:    3 options → 0 = still, 1 = right,    2 = left
        # Shoot:   2 options → 0 = still, 1 = shoot

        self.action_space = spaces.MultiDiscrete([
            3, 3, 2  
        ])
        
    def set_reward_function(self, reward_function: Callable[[dict, dict, int], int]) -> None:
        self.reward_function = reward_function
    
    def set_stage(self, stage: int) -> None:
        """Set the terrain stage"""
        self.game.terrain.set_stage(stage)
        
    def reset(self, seed=None, options=None):
        # We dont use the seed
        if seed is not None:
            np.random.seed(seed)
        
        self.static_counter = 0
        self.curr_obs = None

        self.game.reset(self.random_start)
        obs = OBSParser.get_obs(self.game.get_info())

        enemies = os.path.join(os.getcwd(), "enemy_model/")
        
        self._load_random_enemy_model(enemies)

        return obs, {} 

    def _load_random_enemy_model(self, enemies):
        if os.path.exists(enemies) and len(os.listdir(enemies)) > 0:
            enemy_models = [f for f in os.listdir(enemies) if f.endswith('.zip')]
            if len(enemy_models) > 0:
                random_enemy_model = np.random.choice(enemy_models)
                enemy_model = os.path.join(enemies, random_enemy_model)
                self.game.blue_player.set_model(
                    PPO.load(enemy_model)
                )
    
    def step(self, action):
        if self.curr_obs == None:
            self.curr_obs = OBSParser.get_obs(state=self.game.get_info())
        reward, terminated, truncated = self.__step(action)
        prev_static_counter = self.static_counter
        self.update_flag = self.__set_counter_to_next_alive()
        self.blue_action = self.game.blue_player.take_action(self.curr_obs) if self.update_flag else None
        
        done = terminated or truncated
        if done:
            winner = self.game.winner()
            self.episode_results["total"] += 1
            if winner == "Red":
                self.episode_results["red_wins"] += 1
            elif winner == "Blue":
                self.episode_results["blue_wins"] += 1
            elif winner == "Draw":
                self.episode_results["draws"] += 1
        
        info = {
            "update_flag": self.update_flag, 
            "Prev main tank": prev_static_counter,
            "Next main tank": self.static_counter, 
            "current tanks": OBSParser.parse_obs(OBSParser.get_obs(state=self.game.get_info()))["own_tanks"], 
            "action": action,
            "episode_results": self.episode_results.copy(),
            "done": done
        }
        
        assert self.curr_obs["own_tanks"][self.static_counter][0] == 1 or (terminated or truncated), "The main tank must be alive unless the game is ended."
        if self.rendering and self.update_flag:
            self.renderer.render_observation(self.curr_obs, self.game.terrain.get_info()["walls"], {"Reward": reward})

        return self.__swap_main_tank(self.static_counter), reward, terminated, truncated, info

    def __set_counter_to_next_alive(self):
        update_flag = False
        while True:
            self.static_counter += 1
            if self.static_counter > 1:
                self.static_counter = 0
                self.curr_obs = OBSParser.get_obs(state=self.game.get_info())
                update_flag = True
                self.game.flag_update()
                if sum(tank[0] for tank in self.curr_obs["own_tanks"]) == 0:
                    break
            if self.curr_obs["own_tanks"][self.static_counter][0] == 1:
                break
        return update_flag

    def __step(self, action):
        self.game.step(red_action=action, 
                        blue_action= self.blue_action,
                        static_counter=self.static_counter,
                        update_flag=self.update_flag
                        )
        reward = self.reward_function(
            self.game.red_player.sweet_swap(
                OBSParser.parse_obs(
                    OBSParser.get_obs(state=self.game.get_info())), self.static_counter), # Take a accumulated expectated state
                                      self.game.terrain.get_info(),
                                      self.game.time_counter)
        terminated = self.game.is_game_ended()
        truncated = self.game.time_finished()
        return reward, terminated, truncated 

    def __swap_main_tank(self, i):
        return self.__swap_tank(i, 0)
    
    def __swap_tank(self, i, j):
        return self.game.red_player.sweet_swap(self.curr_obs, i)
        
    def _get_obs(self)->np.array:
        return OBSParser.get_obs(self.game.get_info())
    
    def get_episode_results(self):
        return self.episode_results.copy()
    
    def reset_episode_results(self):
        self.episode_results = {"red_wins": 0, "blue_wins": 0, "draws": 0, "total": 0}

