import sys
import random
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm
import multiprocessing as mp
import csv_utils
import math

from deliveryEnvironment.index import *
from qlearning.agent.deliveryQAgent import *
from qlearning.run import *
from utils.calc import *

plt.style.use("seaborn-v0_8-dark")

sys.path.append("../")

# 待完成事項
# 1. 須建立 tree (或是 k-means) 決定，感測器的回傳sink的資料傳輸路徑?
# 2 判斷感測器是否為隔離節點的方法，利用 Dijkstra’s 決定節點的傳回sink的路徑，如果沒有回傳路徑則為孤立節點
# 若孤立節點附近有可連通節點，則將這些節點加入成一個區塊，為孤立區域

# 使用 2opt 優化 跑UAV加入的節點路徑
# 

# 需要確認 起始點位置，是否在角落

# 使用 batch 改善 Q learning

# 設定環境參數
num_processes = 1 # 使用的多核數量 (產生結果數量)
num_points = 100 # 節點數

n_episodes = 2000 # 訓練次數
num_uav_loops = 1 # UAV 拜訪幾輪


def getMinDistancePoint(env, curr_point):
    min_distance = float('inf')
    min_point = None

    for point in env.unvisited_stops:

        point_x = env.x[[point, curr_point]]
        point_y = env.y[[point, curr_point]]

        distance = calcDistance(point_x, point_y)
        if distance < min_distance:
            min_distance = distance
            min_point = point

    return min_point

# 計算 UAV探索飄移節點需要花費的電量
def calc_drift_cost(position_x, position_y, env):
    drift_distance = calcDistance(position_x, position_y)

    if drift_distance <= env.point_range:
        return 0
    else:
        return env.drift_max_cost


def run_uav(env, init_position):

    idx = 0

    while idx < len(env.stops):

        route = env.stops[idx]

        init_pos_x, init_pos_y = init_position

        position_x = env.x[env.stops]
        position_y = env.y[env.stops]

        # 搜尋飄移節點 所需要的電量消耗
        drift_cost = calc_drift_cost(
            [init_pos_x[route],  position_x[idx]], 
            [init_pos_y[route],  position_y[idx]], 
            env
        )

        env.drift_cost_list[idx] = drift_cost

        drift_remain_cost = env.drift_max_cost - drift_cost
        # 用於搜尋飄移節點中的剩餘能量
        env.remain_power = env.remain_power + drift_remain_cost

        point = getMinDistancePoint(env, idx)

        # 還有剩餘電量加入新的節點
        if point is not None:
            oldStops = list(env.stops)
            oldDrift = list(env.drift_cost_list)

            env.stops.insert(idx + 1, point)
            env.drift_cost_list.insert(idx + 1, env.drift_max_cost)

            new_cost = calcPowerCost(env)

            # 大於能量消耗 就還原節點
            if new_cost > env.max_move_distance:
                env.stops = list(oldStops)
                env.drift_cost_list = list(oldDrift)

            if new_cost <= env.max_move_distance:
                print('113', env.stops)
                env.remain_power = new_cost

        idx = idx + 1

    
    print('117', env.stops)
    distance = calcDistance(env.x[env.stops], env.y[env.stops])
    route,cost = optimal_route(env.stops, env, distance)
    startIndex = calcNodeToOriginDistance(env)
    env.stops = route[startIndex:] + route[:startIndex]
    print('122', env.stops)

    return env

def runMain(index):
    print(f'run {index} start ========================================')
    
    parmas_arr = [
        { "epsilon_min": 0.05, "gamma": 0.1 },
        { "epsilon_min": 0.05, "gamma": 0.2 },
        { "epsilon_min": 0.05, "gamma": 0.3 },
        { "epsilon_min": 0.05, "gamma": 0.4 },
        { "epsilon_min": 0.05, "gamma": 0.5 },
        { "epsilon_min": 0.05, "gamma": 0.6 },
        { "epsilon_min": 0.05, "gamma": 0.7 },
        { "epsilon_min": 0.05, "gamma": 0.8 },
        { "epsilon_min": 0.05, "gamma": 0.9 },
        { "epsilon_min": 0.05, "gamma": 1.0 },

    ]
    
    for params in parmas_arr:

        env = DeliveryEnvironment(num_points, num_points)
        # 感測器初始座標 (水下定錨座標)
        init_X = np.array(env.x)
        init_Y = np.array(env.y)
        init_position = [init_X, init_Y]

        agent = DeliveryQAgent(
                states_size=num_points,
                actions_size=num_points,
                epsilon = 1.0,
                epsilon_min = params["epsilon_min"],
                epsilon_decay = 0.9998,
                gamma = params["gamma"],
                lr = 0.65
            )

        for num in range(num_uav_loops):

            # 跑 Q learning
            env,agent = run_n_episodes(
                env, 
                agent,
                n_episodes=n_episodes,
                result_index=index,
                loop_index=num+1,
                train_params=params,
            )

            if env.stops == []:
                print('no stops')
                break
                
            # uav 開始飛行
            env = run_uav(env, init_position)

            # 產生UAV路徑圖
            uav_run_img = env.render(return_img = True)
            imageio.mimsave(f"./result/{index}_epsilon_min{params['epsilon_min']}_gamma{params['gamma']}_loop_index{num+1}_UAV_result.gif",[uav_run_img],fps = 10)

            # 清除無人跡拜訪後的感測器資料
            env.clear_data()

            # 判斷是否為孤立節點 (需飄移完成再判斷?)
            env.set_isolated_node()

            env.x = np.array(init_X)
            env.y = np.array(init_Y)

            # 執行節點飄移
            env.drift_node()
            env.generate_data()
    print(f'run {index} end ========================================')

# mutiprocessing start ================================
if __name__ == '__main__':
    process_list = []
    csv_utils.write('./result/train_table.csv', 
        [['red_distance','q_distance','opt_distance']]
    )

    for i in range(num_processes):
        process_list.append(mp.Process(target=runMain, args=(i,)))
        process_list[i].start()

    for i in range(num_processes):
        process_list[i].join()

# mutiprocessing end ================================