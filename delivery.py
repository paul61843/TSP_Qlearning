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
import copy

from deliveryEnvironment.index import *
from qlearning.agent.deliveryQAgent import *
from qlearning.run import *
from utils.calc import *

from method.greedy_UAV import *
from method.method1 import *
from method.method2 import *


plt.style.use("seaborn-v0_8-dark")

sys.path.append("../")

# 待完成事項

# 獎勵值 可以減去未拜訪的節點數

# 使用 batch 改善 Q learning

# 比較對象
# 1.	單純的multi-hop transmission
# 2.	UAV拜訪所有匯聚點（不考慮飄浮）
# 3.	Multi-hop transmission + UAV拜訪無法透過multi-hop transmission回傳資料的匯聚點（不考慮飄浮）
# 4.	Multi-hop transmission + UAV拜訪無法透過multi-hop transmission回傳資料的匯聚點（考慮飄浮，以繞圓形的方式拜訪匯聚點之飄浮範圍）
# 5.	我們的方法1：Multi-hop transmission + UAV拜訪無法透過multi-hop transmission回傳資料的匯聚點（考慮飄浮，有飄浮節點最後的gps座標資訊，以繞扇形的方式拜訪匯聚點之飄浮範圍）
# 6.	我們的方法2：將方法1納入load-balance考量（i.e. 納入電量消耗較大的節點做為拜訪點）

# Q learning 參數
# 要使用 基因演算法 比較有依據

# 設定環境參數
num_processes = 1 # 同時執行數量 (產生結果數量)
num_points = 100 # 節點數
max_box = 1000 # 場景大小

n_episodes = 2000 # 訓練次數
num_uav_loops = 10 # UAV 拜訪幾輪

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
        env.unvisited_stops = env.get_unvisited_stops()

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
                env.remain_power = new_cost

        idx = idx + 1

    
    distance = calcDistance(env.x[env.stops], env.y[env.stops])
    route,cost = optimal_route(env.stops, env, distance)
    startIndex = env.first_point
    env.stops = route[startIndex:] + route[:startIndex]

    return env

def runMain(index):
    print(f'run {index} start ========================================')
    
    parmas_arr = [
        # { "epsilon_min": 0.05, "gamma": 0.62, "lr": 0.60 },
        # { "epsilon_min": 0.05, "gamma": 0.62, "lr": 0.61 },
        # { "epsilon_min": 0.05, "gamma": 0.62, "lr": 0.62 },
        # { "epsilon_min": 0.05, "gamma": 0.62, "lr": 0.63 },
        # { "epsilon_min": 0.05, "gamma": 0.62, "lr": 0.64 },
        
        # 好像不錯
        { "epsilon_min": 0.05, "gamma": 0.62, "lr": 0.65 },

        # { "epsilon_min": 0.05, "gamma": 0.62, "lr": 0.67 },
        # { "epsilon_min": 0.05, "gamma": 0.62, "lr": 0.68 },
        # { "epsilon_min": 0.05, "gamma": 0.62, "lr": 0.69 },
        # { "epsilon_min": 0.05, "gamma": 0.62, "lr": 0.70 },
    ]
    
    for params in parmas_arr:

        env = DeliveryEnvironment(num_points, num_points)

        env_mutihop = copy.deepcopy(env)
        env_greedy = copy.deepcopy(env)
        env_greedy_not_drift = copy.deepcopy(env)
        env_Q = copy.deepcopy(env)

        # 感測器初始座標 (水下定錨座標)
        init_X = np.array(env.x)
        init_Y = np.array(env.y)
        init_position = [init_X, init_Y]

        for num in range(num_uav_loops):
            env_mutihop.x = np.array(init_X)
            env_mutihop.y = np.array(init_Y)
            env_greedy.x = np.array(env.x)
            env_greedy.y = np.array(env.y)
            env_greedy_not_drift.x = np.array(init_X)
            env_greedy_not_drift.y = np.array(init_Y)
            env_Q.x = np.array(env.x)
            env_Q.y = np.array(env.y)

            # # =============== Q learning ===============

            # agent = DeliveryQAgent(
            #     states_size=num_points,
            #     actions_size=num_points,
            #     epsilon = 1.0,
            #     epsilon_min = params["epsilon_min"],
            #     epsilon_decay = 0.9998,
            #     gamma = params["gamma"],
            #     lr = params["lr"]
            # )

            # # 跑 Q learning
            # env_Q,agent = run_n_episodes(
            #     env_Q, 
            #     agent,
            #     n_episodes=n_episodes,
            #     result_index=index,
            #     loop_index=num+1,
            #     train_params=params,
            # )

            # # 產生UAV路徑圖
            # uav_run_img = env_Q.render(return_img = True)
            # imageio.mimsave(f"./result/{index}_epsilon_min{params['epsilon_min']}_gamma{params['gamma']}_lr{params['lr']}_loop_index{num+1}_UAV_result.gif",[uav_run_img],fps = 10)

            # # # uav 開始飛行
            # env = run_uav(env, init_position)

            # # =============== Q learning end ===========

            if env.stops == []:
                print('no stops')
                break

            # =============== greedy ===============

            # 跑 uav greedy
            env_greedy = run_n_greedy(
                env_greedy, 
                n_episodes=n_episodes,
                result_index=index,
                loop_index=num+1,
                train_params=params,
            )

            # 產生UAV路徑圖
            uav_run_img = env_greedy.render(return_img = True)
            imageio.mimsave(f"./result/greedy/{index}_loop_index{num+1}_UAV_result.gif",[uav_run_img],fps = 10)

            # =============== greedy end ===========

            # =============== greedy not drift ===============
            # 跑 uav greedy
            env_greedy_not_drift = run_n_greedy(
                env_greedy_not_drift, 
                n_episodes=n_episodes,
                result_index=index,
                loop_index=num+1,
                train_params=params,
            )

            # 產生UAV路徑圖
            uav_run_img = env_greedy_not_drift.render(return_img = True)
            imageio.mimsave(f"./result/greedy_not_drift/{index}_loop_index{num+1}_UAV_result.gif",[uav_run_img],fps = 10)

            # =============== greedy  not drift end ===============

            # =============== mutihop ===============
            uav_run_img = env_mutihop.render(return_img = True)
            imageio.mimsave(f"./result/only_mutihop/{index}_loop_index{num+1}_UAV_result.gif",[uav_run_img],fps = 10)
            # =============== mutihop end ===============


            # 清除無人跡拜訪後的感測器資料
            env.clear_data()
            env_greedy.clear_data()
            env_greedy_not_drift.clear_data()

            env.x = np.array(init_X)
            env.y = np.array(init_Y)

            # 執行節點飄移
            env.drift_node()

            # 隨機產生資料
            add_data = np.random.randint(env.data_generatation_range, size=env.max_box)

            env.generate_data(add_data)
            env_mutihop.generate_data(add_data)
            env_Q.generate_data(add_data)
            env_greedy.generate_data(add_data)
            env_greedy_not_drift.generate_data(add_data)

            # 判斷是否為孤立節點
            env.set_isolated_node()
            
            # 感測器儲存的資料，減去mutihop幫傳的資料
            env_mutihop.subtract_mutihop_data()
            env_greedy.subtract_mutihop_data()
            env_greedy_not_drift.subtract_mutihop_data()
            env_Q.subtract_mutihop_data()

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