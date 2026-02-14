import numpy as np
from stable_baselines3 import PPO

from src.terrain.terrain import Terrain
from src.player import Player
from src.ObservationParser import OBSParser

class Game:
    red_player: Player
    blue_player: Player
    terrain: Terrain
    title: str
    time_limit: int
    
    logger: any
    
    time_counter = 0

    def __init__(self, logger=None, time_limit: int = 12000):
        width = 1920 / 2
        height = 1080 / 2
        self.logger = logger
        self.time_limit = time_limit
        

        #Crear objetos
        self.terrain = Terrain(window_height=height, window_width=width)

        self.red_player = Player(team="Red", terrain=self.terrain)
        self.blue_player = Player(team="Blue", terrain=self.terrain)
        self.red_player.set_enemy_tanks(self.blue_player.get_tanks())
        self.blue_player.set_enemy_tanks(self.red_player.get_tanks())

    def time_finished(self) -> int:
        # this function only returns if the match finished by time limit, not who wins
        return self.time_counter >= self.time_limit 
    
    def is_game_ended(self) -> int:
        # this function only returns if the match has naturally ended, not who wins
        # True == Ended, False == Not Ended
        info = self.get_info()
        red_tanks = info["red_team"]["perceiving_state"]["tanks"]
        blue_tanks = info["red_team"]["perceiving_state"]["enemy_tanks"]
            
        red_state = False
        blue_state = False
            
        for i in range(2):
            red_state = red_state or red_tanks[i]["alive"]
            blue_state = blue_state or blue_tanks[i]["alive"]
            
        return not (red_state and blue_state)
    
    def winner(self) -> str:
        ''' Returns "Red", "Blue", "Draw", or "Undecided" '''
        info = self.get_info()
        red_tanks = info["red_team"]["perceiving_state"]["tanks"]
        blue_tanks = info["red_team"]["perceiving_state"]["enemy_tanks"]
        
        red_alive = sum(tank["alive"] for tank in red_tanks)
        blue_alive = sum(tank["alive"] for tank in blue_tanks)
        
        if red_alive > blue_alive:
            return "Red"
        elif blue_alive > red_alive:
            return "Blue"
        elif red_alive == blue_alive:
            if self.time_finished():
                # The team with alive tanks that are most near the center of the map wins
                red_center = sum(tank["x"] for tank in red_tanks if tank["alive"]) / red_alive, sum(tank["y"] for tank in red_tanks if tank["alive"]) / red_alive
                blue_center = sum(tank["x"] for tank in blue_tanks if tank["alive"]) / blue_alive, sum(tank["y"] for tank in blue_tanks if tank["alive"]) / blue_alive
                red_distance = ((red_center[0] - 1920 / 4) ** 2 + (red_center[1] - 1080 / 4) ** 2)
                blue_distance = ((blue_center[0] - 1920 / 4) ** 2 + (blue_center[1] - 1080 / 4) ** 2)
                if red_distance < blue_distance:
                    return "Red"
                elif blue_distance < red_distance:
                    return "Blue"
                else:
                    return "Draw"
            else:
                return "Undecided"
        
    def step(self, red_action, blue_action, static_counter: int, update_flag: bool) -> None:
        
        self.time_counter += 1

        self.red_player.set_inputs(red_action, static_counter)
        if update_flag:
            self.blue_player.set_batched_inputs(blue_action)
    
    def set_models(self, red_model: PPO, blue_model: PPO) -> None:
        self.red_player.set_model(red_model)
        self.blue_player.set_model(blue_model)

    def debug_step(self, red_action, blue_action) -> tuple[np.array, bool, int]:
        self.time_counter += 1
        self.red_player.set_batched_inputs(red_action)
        self.blue_player.set_batched_inputs(blue_action)
        return OBSParser.get_obs(self.get_info()), self.is_game_ended() or self.time_finished(), self.time_counter

    def flag_update(self) -> None:
        self.red_player.check_alive_tanks()
        self.blue_player.check_alive_tanks()
        
    def get_info(self) -> dict:
        return {
            "red_team": self.red_player.get_game_state(),
            "blue_team": self.blue_player.get_game_state()
        }
    
    def reset(self, random_start: bool) -> np.array:
        self.time_counter = 0
        if random_start:
            new_positions = self.terrain.get_spawn_positions()
            for i in range(2):
                self.red_player.init_state["own_tanks"][i][2] = new_positions[i][0]
                self.red_player.init_state["own_tanks"][i][3] = new_positions[i][1]

            new_positions = self.terrain.get_spawn_positions()
            for i in range(2):
                self.blue_player.init_state["enemy_tanks"][i][2] = 1920 / 2 - new_positions[i][0] 
                self.blue_player.init_state["enemy_tanks"][i][3] = new_positions[i][1]
        self.red_player.reset()
        self.blue_player.reset()
        return OBSParser.get_obs(self.get_info(), team="red")
