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

plt.style.use("seaborn-v0_8-dark")

sys.path.append("../")

# 待完成事項
# 1. 須建立 tree (或是 k-means) 決定，感測器的回傳sink的資料傳輸路徑?
# 2 判斷感測器是否為隔離節點的方法，利用 Dijkstra’s 決定節點的傳回sink的路徑，如果沒有回傳路徑則為孤立節點
# 若孤立節點附近有可連通節點，則將這些節點加入成一個區塊，為孤立區域
# 3. 跑 Q learning 計算能量消耗，需加上探索飄移節點花費的電量

# ===== 4.5 可能做完完了，需測試
# 4. 跑到每一個點後(必做)，需計算無人機剩餘的電量，決定是否添加新的拜訪點
# 5. 飄移後的節點，因離開原始位置，無人機需增加搜尋功能，找尋漂離的節點
# 

# 設定環境參數
num_processes = 1 # 使用的多核數量 (產生結果數量)
num_points = 50 # 節點數

n_episodes = 10 # 訓練次數
num_uav_loops = 1 # UAV 拜訪幾輪


def calcDistance(x, y):
    distance = 0
    for i in range(len(x) - 1):
        distance += np.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2)
    
    return distance

# 計算移動距離，是否超過最大限制
def calcRouteDistance(env):
    cost = calcDistance(env.x[env.stops], env.y[env.stops])

    to_start_distance = 0
    if cost >= 1:
        to_start_distance = calcDistance(env.x[[env.stops[0], env.stops[-1]]], env.y[[env.stops[0], env.stops[-1]]])
        
    cost += to_start_distance

    return cost

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


def run_uav_path(env, init_position):

    curr_cost = calcRouteDistance(env)

    # uav 剩餘的電量 (還需要減去 每次保留的飄移節點值)
    remain_cost = env.max_move_distance - curr_cost

    for idx, route in enumerate(env.stops):

        init_pos_x, init_pos_y = init_position

        position_x = env.x[env.stops]
        position_y = env.y[env.stops]

        # 搜尋飄移節點 所需要的電量消耗 (目前是計算直線距離)
        drift_cost = calc_drift_cost(
            [init_pos_x[idx],  position_x[idx]], 
            [init_pos_y[idx],  position_y[idx]], 
            env
        )

        # 
        drift_remain_cost = env.drift_max_cost - drift_cost
        # 用於搜尋飄移節點中的剩餘能量
        remain_cost = remain_cost + drift_remain_cost

        point = getMinDistancePoint(env, idx)

        # 還有剩餘電量加入新的節點
        if point is not None:
            nextPoint = env.stops[-1] if idx == len(env.stops) - 1 else env.stops[idx + 1]
            position_x = env.x[[idx, point, nextPoint]]
            position_y = env.y[[idx, point, nextPoint]]

            cost = calcDistance(position_x, position_y) + env.drift_max_cost
            if cost < remain_cost:
                env.stops.insert(idx + 1, point)
                env.unvisited_stops.remove(point)
                remain_cost = remain_cost - cost

    return env

# 計算 UAV探索飄移節點需要花費的電量
def calc_drift_cost(position_x, position_y, env):
    dript_distance = calcDistance(position_x, position_y)
    print('543', position_x, position_y, dript_distance)

    if dript_distance <= env.point_range:
        return 0
    else:
        return env.drift_max_cost

def runMain(index):
    print(f'run {index} start ========================================')
    
    parmas_arr = [
        # { "epsilon_min": 0.01 },
        # { "epsilon_min": 0.02 },
        # { "epsilon_min": 0.03 },
        # { "epsilon_min": 0.04 },
        { "epsilon_min": 0.05 },
        # { "epsilon_min": 0.06 },
        # { "epsilon_min": 0.07 },
        # { "epsilon_min": 0.08 },
        # { "epsilon_min": 0.09 },
        # { "epsilon_min": 0.1 },
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
                gamma = 0.65,
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
            
            # reset Q learning
            env,agent,episode_reward = run_episode(env,agent,verbose = 0)

            # uav 開始飛行
            env = run_uav_path(env, init_position)

            # 產生路徑圖
            env.render(return_img = True)

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