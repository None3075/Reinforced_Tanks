import json
import pygame
import numpy as np
from typing import Optional, Dict, Any
from hyperparameters import REWARD_FUNCTIONS
import os
import cv2

class GameRenderer:
    """
    Standalone renderer that visualizes game states.
    Can render observations from any source (environment, file, network, etc.)
    """
    
    def __init__(self, width: int = 960, height: int = 540, fps: int = 30):
        pygame.init()
        self.width = width
        self.height = height
        self.fps = fps
        self.tile_size = 128
        
        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("ReinforcedTanks Viewer")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.WHITE = (255, 255, 255)
        
        # Load background and sprites
        try:
            self.background = pygame.image.load(os.path.join("assets", "reduced.png")).convert_alpha()
            default_rect = self.background.get_rect()
            self.terrain_proportion_x = width / default_rect.width
            self.terrain_proportion_y = height / default_rect.height
            self.background = pygame.transform.scale(self.background, (width, height))
            self.load_tank_sprites()
            self.load_weapon_sprites()
            
        except:
            raise RuntimeError("Failed to load assets. Make sure the 'assets' folder is present.")

        # Font for displaying info
        self.font = pygame.font.SysFont(None, 24)
        
    def render_observation(self, obs: Dict[str, Any], terrain_info: Optional[Dict] = None, 
                          extra_info: Optional[Dict] = None) -> bool:
        """
        Render a game observation.
        
        Args:
            obs: Observation dictionary from the environment
            terrain_info: Optional terrain information
            extra_info: Optional extra information to display (rewards, etc.)
            
        Returns:
            bool: False if window was closed, True otherwise
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        #Clear screen
        if self.background:
            self.window.blit(self.background, (0, 0))
        else:
            self.window.fill(self.BLACK)
        
        # Draw terrain
        # if terrain_info:
        #     self._draw_terrain(terrain_info)

        # Parse observation
        parsed_obs = obs
        # Draw game elements
        self._draw_tanks(parsed_obs["own_tanks"], self.RED, "Red")
        self._draw_bullets(parsed_obs["own_bullets"], self.RED)
        self._draw_tanks(parsed_obs["enemy_tanks"], self.BLUE, "Blue")
        self._draw_bullets(parsed_obs["enemy_bullets"], self.BLUE)
        
        # Draw extra information
        if extra_info:
            self._draw_info(extra_info)
        
        pygame.display.flip()
        self.clock.tick(self.fps)
        
        return True
    
    def load_tank_sprites(self):
        # Load scales
        tank_width = self.tile_size * self.terrain_proportion_x * 2
        tank_height = self.tile_size * self.terrain_proportion_y * 2
        
        # Load tank frames
        self.red_tank_frames = []
        frames_paths = os.listdir(os.path.join("assets/processed", "red", "body"))
        for file in frames_paths:
            if file.endswith(".png"):
                default_frame = pygame.image.load(os.path.join("assets/processed", "red", "body", file)).convert_alpha()
                rotated_frame = pygame.transform.rotate(default_frame, 90)
                frame = pygame.transform.scale(rotated_frame, (tank_width, tank_height))
                self.red_tank_frames.append(frame)
        self.blue_tank_frames = []
        frames_paths = os.listdir(os.path.join("assets/processed", "blue", "body"))
        for file in frames_paths:
            if file.endswith(".png"):
                default_frame = pygame.image.load(os.path.join("assets/processed", "blue", "body", file)).convert_alpha()
                rotated_frame = pygame.transform.rotate(default_frame, 90)
                frame = pygame.transform.scale(rotated_frame, (tank_width, tank_height))
                self.blue_tank_frames.append(frame)

        # Calculate proportions and radius
        self.tank_proportion_x = tank_width / rotated_frame.get_width()
        self.tank_proportion_y = tank_height / rotated_frame.get_height()

        self.tank_radius = tank_width // 3
        self.tank_iterators = [0, 0, 0, 0, 0, 0] 
        
    def load_weapon_sprites(self):
        image = pygame.image.load(os.path.join("assets", "processed", "red", "weapons", "weapon000.png")).convert_alpha()
        image = pygame.transform.rotate(image, -90)
        image = pygame.transform.scale(image, (int(image.get_width() * self.tank_proportion_x), int(image.get_height() * self.tank_proportion_y)))
        self.red_weapon_sprite = image

        image = pygame.image.load(os.path.join("assets", "processed", "blue", "weapons", "weapon000.png")).convert_alpha()
        image = pygame.transform.rotate(image, -90)
        image = pygame.transform.scale(image, (int(image.get_width() * self.tank_proportion_x), int(image.get_height() * self.tank_proportion_y)))
        self.blue_weapon_sprite = image

    def _draw_terrain(self, terrain_info: Dict):
        """Draw terrain walls"""
        # Handle both list format and dict format with 'walls' key

        walls = terrain_info
        
        for wall_info in walls:
            # Extract wall arguments (handles nested structure)
            wall_args = wall_info.get("args", wall_info)
            top_left = wall_args["top_left"]
            bottom_right = wall_args["bottom_right"]
            
            rect = pygame.Rect(
                top_left[0], top_left[1],
                bottom_right[0] - top_left[0],
                bottom_right[1] - top_left[1]
            )
            pygame.draw.rect(self.window, self.GREEN, rect, 3)
    
    def _draw_tanks(self, tanks: list, color: tuple, team: str):
        """Draw tanks from observation data"""
        # Indices 0 alive, 1 angle, 2 x, 3 y
        for i, tank in enumerate(tanks):
            if tank[0] == 0:
                continue
            # Rotate the current frame based on the angle
            if team == "Red":
                frame_rotated = pygame.transform.rotate(self.red_tank_frames[self.tank_iterators[i]], -tank[1])
            else:
                frame_rotated = pygame.transform.rotate(self.blue_tank_frames[self.tank_iterators[i]], -tank[1])

            rotated_frame_rect = frame_rotated.get_rect(center=(int(tank[2]), int(tank[3])))

            # Draw the rotated frame on the window
            self.window.blit(frame_rotated, (rotated_frame_rect.x, rotated_frame_rect.y))

            # Draw the tank hitbox
            pygame.draw.circle(self.window, color, (int(tank[2]), int(tank[3])), self.tank_radius, 2)

            # Update the iterator to cycle through frames
            if team == "Red":
                self.tank_iterators[i] = (self.tank_iterators[i] + 1) % len(self.red_tank_frames)
            else:
                self.tank_iterators[i] = (self.tank_iterators[i] + 1) % len(self.blue_tank_frames)

            # Draw the weapon
            if team == "Red":
                rotated_image = pygame.transform.rotate(self.red_weapon_sprite, -tank[1])
            else:
                rotated_image = pygame.transform.rotate(self.blue_weapon_sprite, -tank[1])
            rotated_rect = rotated_image.get_rect(center=(int(tank[2]), int(tank[3])))
            self.window.blit(rotated_image, (rotated_rect.x, rotated_rect.y))
  
    def _draw_bullets(self, bullets: list, color: tuple):
        """Draw bullets from observation data"""
        for bullet in bullets:
            alive, angle, bounces, x, y = bullet
            if alive > 0.5:
                pygame.draw.circle(self.window, color, (int(x), int(y)), 5)
                
                # Draw bounce count near bullet
                if bounces > 0:
                    text = self.font.render(str(int(bounces)), True, self.BLACK)
                    self.window.blit(text, (int(x) + 10, int(y) - 10))
    
    def _draw_info(self, info: Dict):
        """Draw extra information on screen"""
        y_offset = 10
        for key, value in info.items():
            text = self.font.render(f"{key}: {value}", True, self.WHITE)
            
            # Draw black background for text
            bg_rect = text.get_rect()
            bg_rect.topleft = (10, y_offset)
            bg_rect.width += 10
            bg_rect.height += 4
            pygame.draw.rect(self.window, self.BLACK, bg_rect)
            
            self.window.blit(text, (10, y_offset))
            y_offset += 30
    
    def close(self):
        """Clean up pygame resources"""
        pygame.quit()
        print("Pygame quit called.")

    def renderGame(gamePath: str, output_video: str = None):
        """
        Render a game from a sequence of observation files.
        
        Args:
            gamePath: Path to the folder containing observation files
            output_video: Optional path for output MP4 video. If None, defaults to gamePath/game.mp4
        """
        pygame.init()
        renderer = GameRenderer()

        # Setup video writer
        if output_video is None:
            output_video = os.path.join("game.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, renderer.fps, (renderer.width, renderer.height))

        obs_files = os.listdir(gamePath)
        # Sort observation files to ensure correct order
        obs_files.sort()

        obs_files = [os.path.join(gamePath, f"obs_{i}.json") for i in range(len(obs_files)-1)]
        # Read all observations from files and store in a list
        obss = []
        
        for file in obs_files:
            with open(file, "r") as f:
                obss.append(json.load(f))
        
        try:
            with open("src/terrain/walls_info.json", "r") as f:
                    terrain_info = json.load(f)
            timestep = 0
            for obs in obss:
                extra_info = {}
                rewards = []
                for i, reward_function in enumerate(REWARD_FUNCTIONS):
                    rewards.append(reward_function(obs, terrain_info, timestep))
                extra_info = {f"Reward{i+1}": rewards[i] for i in range(len(rewards))}
                if not renderer.render_observation(obs, terrain_info, extra_info):
                    break

                # Capture frame for video
                frame = pygame.surfarray.array3d(renderer.window)
                # Convert from (width, height, 3) to (height, width, 3) and RGB to BGR
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

                timestep += 1
        finally:
            video_writer.release()
            renderer.close()
            print(f"Video saved to: {output_video}")
        
if __name__ == "__main__":
    GameRenderer.renderGame("src/video/models")
