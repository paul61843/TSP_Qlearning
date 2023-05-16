import numpy as np

def calcDistance(x, y):
    distance = 0
    for i in range(len(x) - 1):
        distance += np.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2)
    
    return distance

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