import numpy as np
import math

def calcDistance(x, y):
    distance = 0
    for i in range(len(x) - 1):
        distance += np.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2)
    
    return distance

# 計算 UAV探索飄移節點需要花費的電量
def calc_drift_cost(position_x, position_y, env):
    drift_distance = calcDistance(position_x, position_y)

    if drift_distance <= env.uav_range:
        return 0
    else:
        return env.drift_max_cost

def calcAvg(curr_pos, unvisited_stops, env):
    curr_x = env.x[curr_pos]
    curr_y = env.y[curr_pos]

    distance = 0

    for i in range(len(unvisited_stops)):
        distance += np.sqrt(
            (curr_x - env.x[i]) ** 2 + 
            (curr_y - env.y[i]) ** 2
        )

    avg = distance / len(unvisited_stops)

    return avg

# 計算移動距離，是否超過最大限制
def calcPowerCost(env):
    distance_cost = calcDistance(env.x[env.stops], env.y[env.stops])

    drift_cost = sum(env.drift_cost_list)

    to_start_cost = 0
    if distance_cost > 0:
        x = env.x[[env.stops[0], env.stops[-1]]]
        y = env.y[[env.stops[0], env.stops[-1]]]
        to_start_cost = calcDistance(x, y)
        
    cost = distance_cost + drift_cost + to_start_cost

    return cost
    
def calcNodeToOriginDistance(env):
    distances = []

    for idx, x in enumerate(env.x):
        x = env.x[idx]
        y = env.y[idx]
        distances.append(x ** 2 +  y ** 2)

    min_distance = min(distances)
    min_index = distances.index(min_distance)
    return min_index