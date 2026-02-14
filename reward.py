import json
import math
import numpy as np
from functools import cache

@cache
def distance(point1: tuple, point2: tuple) -> float:
    pos1 = np.array(point1[-2:])
    pos2 = np.array(point2[-2:])
    return np.linalg.norm(pos1 - pos2)

@cache
def distance_to_center(tank: tuple) -> float:
    return distance(tank, (480, 270))

@cache
def _R1(tank: tuple, k: int) -> float:
    dc = distance_to_center(tank)
    return (1 - (dc/550.727)) ** k

def R1(obs: dict, k: int) -> float:
    assert k > 0, "k must be positive"
    current_tank = tuple(obs["own_tanks"][0])
    rt = _R1(current_tank, k)
    re = max ( [_R1( tuple(enemy), k) for enemy in obs["enemy_tanks"] if enemy[0]] )
    return rt - re

@cache
def hit_circle(m: float, n: float, h: float, k: float, r: float) -> float:
    """
    Line: y = m x + n
    Circle: (x - h)^2 + (y - k)^2 = r^2   with center (h, k) and radius r

    Returns:
        discriminant (float) of the resulting quadratic in x.
        Interpretation:
            disc < 0  -> no intersection
            disc = 0  -> tangent
            disc > 0  -> two intersections
    """
    # y = m*x + b line equation
    # (x-h)^2 + (mx+n-k)^2 = r^2
    # => A x^2 + B x + C = 0
    A = 1.0 + m * m
    B = 2.0 * (m * (n - k) - h)
    C = h * h + (n - k) * (n - k) - r * r

    disc = B * B - 4.0 * A * C

    return disc

@cache
def angle_between_points(p1, p2) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = math.atan2(dy, dx)  # rango [-pi, pi]
    ang = math.degrees(ang)  # rango [-180, 180]
    ang = ang % 360          # rango [0, 360)
    return ang

@cache
def angular_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)

def R2(obs: dict, current_bullet: tuple, r: int) -> float:
    assert r > 50, "r must be greater than 50"
    current_bullet = current_bullet

    bonification = 0.0
    if not current_bullet[0]: # bullet not alive, considering that the bullet is missing while you are loosing, we penalize a bit
        bonification -= 0.1

    PP, closest_enemy = min(
        ((distance(current_bullet, tuple(t)), t) for t in obs["enemy_tanks"] if t[0]),
        default=(float("inf"), None),
    )
    enemy_pos = tuple(closest_enemy[-2:])
    ang = current_bullet[1] % 360.0          # [0, 360)
    ang = ((ang + 90.0) % 180.0) - 90.0      # (-90, 90]
    ang = np.clip(ang, -89.0, 89.0)          # evita infinidades
    mb = np.tan(np.deg2rad(ang))
    nb = current_bullet[4] - mb * current_bullet[3]
    # Line equation: y = m*x + b
    PR = abs(mb * enemy_pos[0] - enemy_pos[1] + nb) / np.sqrt(mb**2 + 1)

    # Normalize PP and PR
    PP = 1 - (PP / 1101.5) # (0, 1) using the max diagonal distance of the map
    PR = 1 - (PR / 480) # (0, 1) using the max witdh of the map, since PR is the perpendicular distance

    disc = hit_circle(mb, nb, enemy_pos[0], enemy_pos[1], r)
    angle = angle_between_points(current_bullet[-2:], enemy_pos)
    diff = angular_diff(angle, current_bullet[1])

    if disc > 0 and diff < 15.0:
        return PP * PR + bonification
    else:
        return - (PP * PR) + bonification

def R3(obs: dict, r: int) -> float:
    t1 = - max (R2(obs, tuple(bullet), r) for bullet in obs["enemy_bullets"])
    t2 = - max (R2(obs, tuple(bullet), r) for bullet in obs["own_bullets"])
    return min(t1, t2)

def reward_function0(obs: dict, terrain: dict, time: int)-> float:

    n_t_alive = sum(tank[0] for tank in obs["own_tanks"])
    n_e_alive = sum(tank[0] for tank in obs["enemy_tanks"])

    if n_e_alive == 0:
        return 10
    elif n_t_alive == 0:
        return -10
    
    if n_e_alive == n_t_alive:
        k  = 1
        return R1(obs, k)
    # elif n_e_alive > n_t_alive:
    #     r = 100
    #     return R2(obs, tuple(obs["own_bullets"][0]), r)
    # elif n_t_alive > n_e_alive:
    #     r = 100
    #     return R3(obs, r)
    return (n_t_alive - n_e_alive) / 3
