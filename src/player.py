import pygame
import os
from typing import Literal
from src.tank.tank import Tank
from src.terrain.terrain import Terrain
import json
import pprint
from stable_baselines3 import PPO
from src.ObservationParser import OBSParser
from itertools import chain

TILE_SIZE = 128  # Size of the tile in pixels

# This class encapsulates the agent that controls the tanks in the game.
class Player:

    team: str
    alive_tanks: list[Tank]
    tanks: list[Tank]
    enemy_tanks: list[Tank]
    inputs: list[int] = [0, 0, 0, 0, 0, 0]
    terrain: Terrain

    # For mirroring the teams on the map
    TEAM_OFFSET = TILE_SIZE * 24
    SPACING = 5

    model: PPO = None
    
    def __init__(self, team: Literal["Red", "Blue"], terrain: Terrain):
        self.team = team
        self.terrain = terrain

        self.CENTER_Y = TILE_SIZE * terrain.proportion_y * 7
        self.RED_X = TILE_SIZE * terrain.proportion_x * 2
        self.BLUE_X = self.RED_X + self.TEAM_OFFSET * terrain.proportion_x

        self.alive_tanks = []
        self.tanks = []
        with open("example_state.json", "r") as file:
            self.init_state = json.load(file)
        self.init_fixed_tanks()
    
    def reset_tanks_from_state(self):
        """
        Reset the tanks and bullets based on the provided state dictionary.
        
        Args:
            state: A dictionary containing the state of all tanks and bullets
                {
                    "own_tanks": [[alive, angle, x, y], ...],
                    "own_bullets": [[active, x, y, angle, time], ...],
                    "enemy_tanks": [[alive, angle, x, y], ...],
                    "enemy_bullets": [[active, x, y, angle, time], ...]
                }
        """
        # Reset own tanks
        if self.team == "Red":
            for i in range(2):
                self.tanks[i].alive = True
                self.tanks[i].angle = self.init_state["own_tanks"][i][1]
                self.tanks[i].x = self.init_state["own_tanks"][i][2]
                self.tanks[i].y = self.init_state["own_tanks"][i][3]

                self.tanks[i].bullet.alive = 0
                self.tanks[i].bullet.x = 0
                self.tanks[i].bullet.y = 0
                self.tanks[i].bullet.angle = 0
                self.tanks[i].bullet.bounces = 0
        else:
        # Reset enemy tanks
            for i in range(2):
                self.tanks[i].alive = True
                self.tanks[i].angle = self.init_state["enemy_tanks"][i][1]
                self.tanks[i].x = self.init_state["enemy_tanks"][i][2]
                self.tanks[i].y = self.init_state["enemy_tanks"][i][3]
                self.tanks[i].bullet.alive = 0
                self.tanks[i].bullet.x = 0
                self.tanks[i].bullet.y = 0
                self.tanks[i].bullet.angle = 0
                self.tanks[i].bullet.bounces = 0
        
        # Update alive_tanks list
        self.alive_tanks = [tank for tank in self.tanks if tank.alive]

    def init_fixed_tanks(self):
        x_coord = self.BLUE_X if self.team.lower() == "blue" else self.RED_X

        width = TILE_SIZE * self.terrain.proportion_x * 2
        height = TILE_SIZE * self.terrain.proportion_y * 2

        for offset in [-TILE_SIZE * self.terrain.proportion_y * self.SPACING,
                       TILE_SIZE * self.terrain.proportion_y * self.SPACING]:
            tank = Tank(
                x=x_coord,
                y=self.CENTER_Y + offset,
                width=width,
                height=height,
                team=self.team,
                initial_angle=0 if self.team.lower() == "red" else 180,
                terrain=self.terrain
            )
            self.tanks.append(tank)
            self.alive_tanks.append(tank)
                
    def set_enemy_tanks(self, enemy_tanks: list[Tank]):
        self.enemy_tanks = enemy_tanks
        
    def get_tanks(self) -> list[Tank]:
        return self.tanks
        
    def update(self):
        for tank in self.alive_tanks:
            tank.update(self.terrain.walls, self.alive_tanks + self.enemy_tanks)
            tank.update_bullets(self.terrain.walls, self.alive_tanks + self.enemy_tanks)
            
        self.alive_tanks = [tank for tank in self.alive_tanks if tank.alive]
        
    def manual_update(self):
        self.manage_input()
        self.set_manual_inputs(self.inputs[0:2])
        for tank in self.alive_tanks:
            tank.update(self.terrain.walls, self.alive_tanks + self.enemy_tanks)
            tank.update_bullets(self.terrain.walls, self.alive_tanks + self.enemy_tanks)
        # Filter out dead tanks after updates
        self.alive_tanks = [tank for tank in self.alive_tanks if tank.alive]
        
    def draw(self, window):
        for tank in self.alive_tanks:
            tank.draw(window)
    
    def set_inputs(self, inputs: list[int], static_counter: int = 0):
        if len(inputs) != 3:
            raise ValueError(f"Inputs length must match the number of tank's actions. {len(inputs)} != 3")

        self.tanks[static_counter].inputs = inputs
        self.tanks[static_counter].update(self.terrain.walls, self.alive_tanks + self.enemy_tanks)
        self.tanks[static_counter].update_bullets(self.terrain.walls, self.alive_tanks + self.enemy_tanks)
        self.tanks[static_counter].inputs = [0,0,0]
    
    def set_manual_inputs(self, inputs: list[int]):
        if len(inputs) != 2:
            raise ValueError(f"Inputs length must match the number of tank's actions. {len(inputs)} != 2")

        self.tanks[0].inputs = inputs

    def set_batched_inputs(self, inputs: list[int]):
        if len(inputs) != len(self.tanks)*3:
            raise ValueError("Inputs length must match the number of tanks.")

        index = 0
        for tank in self.tanks:
            tank.inputs = inputs[index * 3:(index + 1) * 3]
            index += 1
        self.update()

    def check_alive_tanks(self):
        self.alive_tanks = [tank for tank in self.alive_tanks if tank.alive]
    
    def get_game_state(self):
        perceiving_state = {
            "tanks": [tank.get_info(self.team) for tank in self.tanks],
            "enemy_tanks": [tank.get_info(self.team) for tank in self.enemy_tanks]
            }
        return {
            "perceiving_state": perceiving_state,
            "terrain": self.terrain.get_info()
        }
    
    def reset(self):
        self.reset_tanks_from_state()

    def set_model(self, model: PPO):
        self.model = model
    
    def take_action(self, obs: dict):
        if self.model != None:
            action0, _states = self.model.predict(obs, deterministic=False)
            action1, _states = self.model.predict(self.sweet_swap(obs,1), deterministic=False) if self.tanks[1].alive else ([0,0,0], None)
            action = list(chain(action0, action1)) 
        else:
            action = [0,0,0,0,0,0]
        return action
    
    @staticmethod
    def sweet_swap(obs, i):
        match i:
            case 0:
                return obs
            case 1:
                return {
                    "own_tanks": [obs["own_tanks"][1], obs["own_tanks"][0]],
                    "own_bullets": [obs["own_bullets"][1], obs["own_bullets"][0]],
                    "enemy_tanks": obs["enemy_tanks"],
                    "enemy_bullets": obs["enemy_bullets"]
                }