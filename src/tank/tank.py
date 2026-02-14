from typing import Literal
import os
import numpy as np
from .bullet import Bullet

VELOCITY = 7  # Default velocity for tank movement
RADIUS_FACTOR = 3  # Factor to calculate the radius of the tank hitbox

class Tank:
    """
    Represents a tank in the game, with attributes for position, size, team, and appearance.
    Use get_position() to get the position of the tank in the game world. This will get you the center of the tank sprite, not the top-left corner.
    """

    # Class init attributes
    x: int
    y: int
    width: int
    height: int
    team: Literal["Red", "Blue"]

    # Class Dinamic attributes
    color: tuple[int, int, int]
    proportion_x: float
    proportion_y: float
    angle: float
    bullet: Bullet

    # Input
    inputs: list[int] = [0, 0, 0]

    # Class default attributes
    asset_path: str = os.path.join("assets", "processed")
    angle_offset = -90
    alive: bool
    
    def __init__(self, x: int, y: int, width: int, height: int, team: Literal["Red", "Blue"], initial_angle, terrain = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.radius = width // RADIUS_FACTOR
        self.team = team
        self.terrain = terrain
        self.alive = True

        self.color = (255, 0, 0) if team.lower() == "red" else (0, 0, 255)
        self.iterator = 0
        self.frames = []
        self.angle = initial_angle % 360

        self.load_frames()
        
        angle_rad = np.radians(self.angle)
        self.movement_vector = np.array((VELOCITY * np.cos(angle_rad), VELOCITY * np.sin(angle_rad)))
        self.move(walls=[], tanks=[]) # Initial move to set correct position

        # Add bullet properties
        self.bullet = Bullet(self)
        
    def shoot(self):
        """Start the bullet if is not alived before."""
        if not self.bullet.alive:
            angle_rad = np.deg2rad(self.angle)
            barrel_length = self.width // 2.5
            start_x = self.x + barrel_length * np.cos(angle_rad)
            start_y = self.y + barrel_length * np.sin(angle_rad)
            
            # Check if bullet spawns on terrain
            if self.terrain:
                for wall in self.terrain.walls:
                    if circle_rect_collision(start_x, start_y, 5, wall):
                        self.take_damage()
                        return
            
            # Create a new bullet
            self.bullet.start(
                x=start_x,
                y=start_y,
                angle=self.angle
            )
        
    def update_bullets(self, walls, tanks):
        """Update all bullets fired by this tank."""
        if self.inputs[2] == 1 and self.alive:
            self.shoot()

        if self.bullet.alive:
            self.bullet.update()
                
            # Check for wall collisions
            self.bullet.check_wall_collision(walls)
                
            # Check for tank collisions
            hit_tank = self.bullet.check_tank_collision(tanks)
            if hit_tank:
                hit_tank.take_damage()

    def load_frames(self):
        self.proportion_x = self.width / 128
        self.proportion_y = self.height / 128

    def update(self, walls, tanks):
        self.move(walls, tanks)
        
    def move(self, walls, tanks):
        previous_x, previous_y = self.x, self.y
        
        if self.inputs[1] == 1:
            self.angle = (self.angle + 5) % 360
        elif self.inputs[1] == 2:
            self.angle = (self.angle - 5) % 360
        
        # Update el movement vector para sobreescribir el valor anterior
        angle_rad = np.radians(self.angle)
        self.movement_vector = np.array((VELOCITY * np.cos(angle_rad), VELOCITY * np.sin(angle_rad)))
        
        if self.inputs[0] == 1:
            self.x += self.movement_vector[0]
            self.y += self.movement_vector[1]
        elif self.inputs[0] == 2:
            self.x -= self.movement_vector[0]
            self.y -= self.movement_vector[1]

        # Check for collisions only if position changed
        if self.x != previous_x or self.y != previous_y:
            # Check for terrain collision
            for wall in walls:
                if circle_rect_collision(self.x, self.y, self.radius, wall):
                    # Reset position if collision detected
                    self.x, self.y = previous_x, previous_y
                    break
            else:  # No wall collision detected, check tank collisions
                for tank in tanks:
                    if tank != self and tank.alive:
                        distance = np.hypot(tank.x - self.x, tank.y - self.y)
                        if distance < (self.radius + tank.radius):
                            # Reset position if collision detected
                            self.x, self.y = previous_x, previous_y
                            break
     
    def get_position(self) -> tuple[int, int]:
        return (self.x - self.width/2, self.y - self.height/2)
    
    def get_info(self, player_team: Literal["Red", "Blue"] = None) -> dict:
        bullet = self.bullet.get_info(player_team)

        if not bullet["alive"]:
            bullet["x"] = self.x
            bullet["y"] = self.y
            bullet["angle"] = self.angle
        
        if player_team == "Red":
            return {
                "class": "Tank",
                "x": self.x,
                "y": self.y,
                "angle": self.angle,
                "bullets": bullet,
                "alive": self.alive
            }
        else:
            # Mirror the coordinates for blue team
            map_width = 1920 / 2
            bullet["angle"] = (180 - bullet["angle"]) % 360
            return {
                "class": "Tank",
                "x": map_width - self.x,
                "y": self.y,
                "angle": (180 - self.angle) % 360,
                "bullets": bullet,
                "alive": self.alive
            }
        
    def take_damage(self):
        self.alive = False
        self.bullet.alive = False

def circle_rect_collision(cx, cy, radius, rect):
    # Find the closest point on the rect to the circle
    closest_x = max(rect.top_left[0], min(cx, rect.bottom_right[0]))
    closest_y = max(rect.top_left[1], min(cy, rect.bottom_right[1]))

    # Calculate distance between circle center and this point
    dx = cx - closest_x
    dy = cy - closest_y

    return (dx * dx + dy * dy) < (radius * radius)

