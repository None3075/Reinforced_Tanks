import pygame

keys: dict[int, bool] = {

    # Red team controls
    pygame.K_UP: False,
    pygame.K_DOWN: False,
    pygame.K_LEFT: False,
    pygame.K_RIGHT: False,
    pygame.K_SPACE: False, #Shoot

    # Blue team controls
    pygame.K_a: False,
    pygame.K_d: False,
    pygame.K_w: False,
    pygame.K_s: False,
    pygame.K_RCTRL: False, #Shoot

    # Red team tank selection
    pygame.K_1: False,
    pygame.K_2: False,
    pygame.K_3: False,
    pygame.K_4: False,
}

def press_key(key: int):
    keys[key] = True

def release_key(key: int):
    keys[key] = False

def get_pressed(key: int) -> bool:
    return keys.get(key, False)