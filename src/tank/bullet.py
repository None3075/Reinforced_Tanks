import numpy as np
from typing import Literal

class Bullet:
    """Represents a bullet fired by a tank that can bounce off walls and damage enemy tanks."""

    x: int
    y: int
    radius: int
    angle: float
    team: Literal["Red", "Blue"]
    speed: int
    bounces: int
    alive: bool
    
    # CONSTANTS
    max_bounces: int = 3 
    speed: int = 9
    radius: int = 5
    def __init__(self, owner):
        self.team = owner.team
        self.color = (255, 0, 0) if owner.team == "Red" else (0, 0, 255)
        self.bounces = 0
        self.alive = False
        self.owner = owner  # Reference to the tank that fired it
        
        # Calculate velocity based on angle
    
    def start(self, x: int, y: int, angle: float):
        self.x = x
        self.y = y
        self.angle = angle
        self.alive = True
        
        rad_angle = np.deg2rad(angle)
        self.velocity = np.array([
            self.speed * np.cos(rad_angle),
            self.speed * np.sin(rad_angle)
        ])
        
    def update(self):
        # Move bullet
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        
        # If we've exceeded max bounces, remove the bullet
        if self.bounces > self.max_bounces:
            self.bounces = 0 # Reset the bounce counter
            self.alive = False
    
    def check_wall_collision(self, walls):
        """Check for collisions with walls and bounce."""
        for wall in walls:
            # Check if bullet collides with wall
            if (wall.top_left[0] <= self.x <= wall.bottom_right[0] and 
                wall.top_left[1] <= self.y <= wall.bottom_right[1]):
                
                # Determine which side of the wall was hit
                left_dist = abs(self.x - wall.top_left[0])
                right_dist = abs(self.x - wall.bottom_right[0])
                top_dist = abs(self.y - wall.top_left[1])
                bottom_dist = abs(self.y - wall.bottom_right[1])
                
                min_dist = min(left_dist, right_dist, top_dist, bottom_dist)
                
                # Bounce based on which side was hit
                if min_dist == left_dist or min_dist == right_dist:
                    self.velocity[0] *= -1  # Horizontal bounce
                else:
                    self.velocity[1] *= -1  # Vertical bounce
                
                # Update angle based on new velocity
                self.angle = np.degrees(np.arctan2(self.velocity[1], self.velocity[0]))
                
                if self.angle < 0:
                    self.angle += 360
                elif self.angle >= 360:
                    self.angle -= 360
                
                # Increment bounce counter
                self.bounces += 1
                return True
        return False
    
    def check_tank_collision(self, tanks):
        """Check for collisions with tanks."""
        for tank in tanks:
            # Check collision using distance formula
            distance = np.sqrt((self.x - tank.x)**2 + (self.y - tank.y)**2)
            if distance <= (tank.width // 3 + self.radius) and tank.alive:
                self.alive = False
                self.bounces = 0 # Reset the bounce counter
                return tank  # Return the tank that was hit
        return None
    
    def get_info(self, player_team: Literal["Red", "Blue"] = None) -> dict:
        """Return bullet information as a dictionary."""
        if not self.alive:
            return {
                "class": "Bullet",
                "x": 0,
                "y": 0,
                "angle": 0,
                "team": self.team,
                "bounces": 0,
                "alive": self.alive
            }
        if player_team == "Red":
            return {
                "class": "Bullet",
                "x": self.x,
                "y": self.y,
                "angle": self.angle,
                "team": self.team,
                "bounces": self.bounces,
                "alive": self.alive
            }
        else:
            # Return mirrored information for the enemy team
            map_width = 1920 / 2
            
            return {
                "class": "Bullet",
                "x": map_width - self.x,
                "y": self.y,
                "angle": (180 - self.angle) % 360,
                "team": self.team,
                "bounces": self.bounces,
                "alive": self.alive
            }
        