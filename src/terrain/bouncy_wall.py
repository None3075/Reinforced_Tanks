from typing import Literal
import os

class BouncyWall:
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]

    def __init__(self, top_left: tuple[int, int], bottom_right: tuple[int, int]):
        self.top_left = top_left
        self.bottom_right = bottom_right
    def get_info(self) -> dict[str, tuple[int, int]]:
        return {
            "class": "BouncyWall",
            "args": {
                "top_left": self.top_left,
                "bottom_right": self.bottom_right
            }
        }
    
    def update(self, window):
        ...