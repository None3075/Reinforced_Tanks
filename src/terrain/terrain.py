import os
import json
import warnings
import random
import numpy as np

from src.terrain.bouncy_wall import BouncyWall

TILE_SIZE = 128

class Terrain:

    asset_path: str = os.path.join("assets")
    proportion_x: int
    proportion_y: int

    walls: list[BouncyWall]
    spawn_zones: list[BouncyWall]
    stages_spawn_zones: list[list[BouncyWall]]
    
    save_file: str = os.path.join("src", "terrain", "walls_info.json")

    def __init__(self, window_width: int, window_height: int):
        self.proportion_x = window_width / 3584
        self.proportion_y = window_height / 1792
        
        self.walls = []
        self.spawn_zones = []
        self.stages = []
        self.stage_index = 0
        self.stages_spawn_zones = []
        self.load_walls_from_json()
        self.create_spawn_zones()
        self.spawn_zones = self.stages_spawn_zones[0]
        
    def load_walls_from_json(self):
        try:
            with open(self.save_file, "r") as f:
                walls_info = json.load(f)
                for wall_info in walls_info:
                    wall = BouncyWall(
                        top_left=(wall_info["args"]["top_left"][0] , 
                                  wall_info["args"]["top_left"][1]), 
                        bottom_right=(wall_info["args"]["bottom_right"][0], 
                                      wall_info["args"]["bottom_right"][1])
                    )
                    self.walls.append(wall)
        except FileNotFoundError:
            self.create_walls()
    
    def add_wall(self, top_left: tuple[int, int], bottom_right: tuple[int, int]):
        wall = BouncyWall(
            top_left=(top_left[0]*self.proportion_x , top_left[1]*self.proportion_y), 
            bottom_right=(bottom_right[0]*self.proportion_x, bottom_right[1]*self.proportion_y)
            )
        self.walls.append(wall)

    def add_spawn_zone(self, top_left: tuple[int, int], bottom_right: tuple[int, int], stage_index: int = -1):
        wall = BouncyWall(
            top_left=(top_left[0]*self.proportion_x , top_left[1]*self.proportion_y), 
            bottom_right=(bottom_right[0]*self.proportion_x, bottom_right[1]*self.proportion_y)
            )
        if stage_index == -1:
            return
        else:
            while len(self.stages_spawn_zones) <= stage_index:
                self.stages_spawn_zones.append([])
            self.stages_spawn_zones[stage_index].append(wall)

    def create_wall(self, top_left: tuple[int, int], bottom_right: tuple[int, int]):
        return BouncyWall(
            top_left=(top_left[0]*self.proportion_x , top_left[1]*self.proportion_y), 
            bottom_right=(bottom_right[0]*self.proportion_x, bottom_right[1]*self.proportion_y)
            )
        
    def update(self):
        ...
    
    def get_info(self) -> dict:
        return {
            "class": "Terrain",
            "walls": self.get_spawn_zones_info() + self.get_walls_info(),
        }
    
    def create_spawn_zones(self):

        # Stage 0 spawn zones
        self.add_spawn_zone(top_left=(1400, 200), bottom_right=(1750, 500), stage_index=0)

        self.add_spawn_zone(top_left=(1400, 1300), bottom_right=(1750, 1600), stage_index=0)

        self.add_spawn_zone(top_left=(1600, 600), bottom_right=(1750, 1200), stage_index=0)

        # Stage 1 spawn zones
        self.add_spawn_zone(top_left=(1400, 200), bottom_right=(1750, 500), stage_index=1)

        self.add_spawn_zone(top_left=(1400, 1300), bottom_right=(1750, 1600), stage_index=1)

        self.add_spawn_zone(top_left=(1600, 600), bottom_right=(1750, 1200), stage_index=1)

        self.add_spawn_zone(top_left=(800, 600), bottom_right=(1300, 1200), stage_index=1)

        # Stage 2 spawn zones
        self.add_spawn_zone(top_left=(1400, 200), bottom_right=(1750, 500), stage_index=2)

        self.add_spawn_zone(top_left=(1400, 1300), bottom_right=(1750, 1600), stage_index=2)

        self.add_spawn_zone(top_left=(1600, 600), bottom_right=(1750, 1200), stage_index=2)

        self.add_spawn_zone(top_left=(800, 600), bottom_right=(1300, 1200), stage_index=2)
        
        self.add_spawn_zone(top_left=(200, 200), bottom_right=(900, 400), stage_index=2)

        self.add_spawn_zone(top_left=(200, 500), bottom_right=(450, 1300), stage_index=2)

        self.add_spawn_zone(top_left=(200, 1400), bottom_right=(900, 1600), stage_index=2)


    def get_spawn_positions(self):
        # Select spawn zones based on stage_index
        lst = [i for i in range(len(self.spawn_zones))]
        random.shuffle(lst)

        top_lefts = np.array([self.spawn_zones[i].top_left for i in lst])
        bottom_rights = np.array([self.spawn_zones[i].bottom_right for i in lst])

        xs = np.random.uniform(top_lefts[:, 0], bottom_rights[:, 0])
        ys = np.random.uniform(top_lefts[:, 1], bottom_rights[:, 1])

        return list(zip(xs, ys))

    def get_walls_info(self) -> list[dict]:
        return [wall.get_info() for wall in self.walls]
    
    def get_spawn_zones_info(self) -> list[dict]:
        return [zone.get_info() for zone in self.spawn_zones]
    
    # Deprecated method to create walls if the JSON file is not found  
    def create_walls(self):
        warnings.warn(
            "create_walls, if this is called, it means that the json file 'walls_info.json' is not found. ",
            category=DeprecationWarning,
            stacklevel=2
        )

        self.add_wall(top_left=(48,48), bottom_right=(86, 1754))  # Left wall
        self.add_wall(top_left=(3504, 48), bottom_right=(3542, 1754))  # right wall

        self.add_wall(top_left=(48, 48), bottom_right=(3542, 86)) # Top wall
        self.add_wall(top_left=(48, 1706), bottom_right=(3542, 1754)) # Bottom wall
        
        with open(self.save_file, "w") as f:
            json.dump([wall.get_info() for wall in self.walls], f)
        
    def set_stage(self, stage: int):
        if stage < 0 or stage >= len(self.stages_spawn_zones):
            raise ValueError(f"Stage {stage} does not exist. Available stages: 0 to {len(self.stages_spawn_zones)-1}")
        
        self.spawn_zones = self.stages_spawn_zones[stage]
        self.stage_index = stage