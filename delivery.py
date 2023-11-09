import sys
import time
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
from constants.constants import *

from method.greedy_UAV import *
from method.NJNP import *
from method.method2 import *


plt.style.use("seaborn-v0_8-dark")

sys.path.append("../")

# 比較對象
# 1.	單純的multi-hop transmission
# 2.	UAV拜訪所有匯聚點（不考慮飄浮）
# 3.	Multi-hop transmission + UAV拜訪無法透過multi-hop transmission回傳資料的匯聚點（不考慮飄浮）
# 4.	Multi-hop transmission + UAV拜訪無法透過multi-hop transmission回傳資料的匯聚點（考慮飄浮，以繞圓形的方式拜訪匯聚點之飄浮範圍）
# 5.	我們的方法1：Multi-hop transmission + UAV拜訪無法透過multi-hop transmission回傳資料的匯聚點（考慮飄浮，有飄浮節點最後的gps座標資訊，以繞扇形的方式拜訪匯聚點之飄浮範圍）
# 6.	我們的方法2：將方法1納入load-balance考量（i.e. 納入電量消耗較大的節點做為拜訪點）

# 爆掉的資料量

# 封包抵達率
# UAV代收的資料量 
# overflow 

# 時間 1000 單位

# 至少六張圖

# 感測器的資料經過計算後 回傳成功 (感測器的資料經過計算後 回傳失敗 (斷路且飛機找不到))
# 感測器的資料來不及計算導致 overflow (飛機找不到 飛機來不及到)

# 節點飄移距離參數改變
# 無人機電池電量 
# 感測器 buffer 大小改變

# 2 x 3 

# 節點飄移 需要飄移特定方向 較遠的位置 飄移速度越快


# 設定環境參數
num_processes = 1 # 同時執行數量 (產生結果數量)
num_points = 400 # 節點數
max_box = 2000  # 場景大小 單位 (1m)

n_episodes = 1000 # 訓練次數

# 比較參數
total_data = 0
sensor_data = 0
uav_data = 0

def getMinDistanceIndex(env, curr_point):
    min_distance = float('inf')
    min_point = None

    for (point, index) in enumerate(env.unvisited_stops):

        point_x = env.x[[point, curr_point]]
        point_y = env.y[[point, curr_point]]

        distance = calcDistance(point_x, point_y)
        if distance < min_distance:
            min_distance = distance
            min_index = index

    return min_index



def run_uav(env, init_position, current_time, process_index):
    
    # 到達最後的節點 返回sink
    if len(env.stops) == env.current_run_index + 1:
        env.stops.append(env.first_point)
    

    env.uav_remain_run_distance = env.uav_remain_run_distance + env.uav_speed * 1 # 每秒新增的距離
    
    # 判斷無人機飛是否抵達下一個節點
    x = env.x[[env.stops[env.current_run_index], env.stops[env.next_point]]]
    y = env.y[[env.stops[env.current_run_index], env.stops[env.next_point]]]
    
    init_pos_x, init_pos_y = init_position
    
    distance = math.ceil(calcDistance(x, y))
    


    if env.uav_remain_run_distance >= distance:
        
        # 搜尋飄移節點 所需要的電量消耗
        drift_cost = calc_drift_cost(
            [init_pos_x[env.stops[env.next_point]],  env.x[env.stops[env.next_point]]], 
            [init_pos_y[env.stops[env.next_point]],  env.y[env.stops[env.next_point]]], 
            env
        )
        
        env.drift_cost_list[env.next_point] = drift_cost
        
        next_point_cost = distance + drift_cost
        
        if env.uav_remain_run_distance >= next_point_cost:
            env.uav_remain_run_distance = env.uav_remain_run_distance - next_point_cost
            env.clear_data_one(init_position, env.stops[env.next_point], True)
        
            # 如果抵達 sink，則 reset 環境
            if env.stops[-1] == env.first_point:
                env.stops = []
                env.stops.append(env.first_point)
                env.current_run_index = 0
                env.next_point = 1
                return env
                            
                
            # 新增下一個節點
            env.current_run_index = env.current_run_index + 1
            env.next_point = env.next_point + 1
            
            
            drift_remain_cost = env.drift_max_cost - drift_cost
            # 用於搜尋飄移節點中的剩餘能量
            env.remain_power = env.remain_power + drift_remain_cost
            env.unvisited_stops = env.get_unvisited_stops()
            
            mostDataOfPoint = getMostDataOfSensor(env)
            

            add_index = getMinDistanceIndex(env, mostDataOfPoint)
            
            # 還有剩餘電量加入新的節點
            if mostDataOfPoint is not None:
                oldStops = list(env.stops)
                oldDrift = list(env.drift_cost_list)

                env.stops.insert(add_index + 1, mostDataOfPoint)
                env.drift_cost_list.insert(add_index + 1, env.drift_max_cost)

                new_cost = calcPowerCost(env)

                # 大於能量消耗 就還原節點
                if new_cost > env.max_move_distance:
                    env.stops = list(oldStops)
                    env.drift_cost_list = list(oldDrift)

                if new_cost <= env.max_move_distance:
                    env.remain_power = env.max_move_distance - new_cost


            # 到達最後的節點 返回sink
            if len(env.stops) == env.current_run_index + 1:
                env.stops.append(env.first_point)
                env.drift_cost_list.append(0)
        
        
    # 紀錄資料
    if current_time % env.unit_time == 0:
        mutihop_data = env.sum_mutihop_data * env.calc_data_compression_ratio
        sensor_data_origin = sum(item['origin'] for item in env.data_amount_list)
        sensor_data_calc = sum(item['calc'] for item in env.data_amount_list) * env.calc_data_compression_ratio
        sensor_data = sensor_data_origin + sensor_data_calc
        
        uav_data_origin = env.uav_data['origin']
        uav_data_calc = env.uav_data['calc'] * env.calc_data_compression_ratio
        uav_data = uav_data_calc +uav_data_origin
        
        total_data = env.generate_data_total
        lost_data = total_data - (mutihop_data + sensor_data + uav_data)
        run_time = current_time
        env.result.append([
            math.ceil(run_time), 
            math.ceil(total_data // 8),
            math.ceil(mutihop_data // 8), 
            math.ceil(sensor_data_origin // 8), 
            math.ceil(sensor_data_calc // 8), 
            math.ceil(sensor_data // 8), 
            math.ceil(uav_data_calc //8), 
            math.ceil(uav_data_origin // 8), 
            math.ceil(uav_data // 8), 
            math.ceil(lost_data // 8),
        ])
        csv_utils.writeDataToCSV(f'./result/csv{process_index}/q_learning.csv', env.result)

    # 產生UAV路徑圖
    if current_time % 1000 == 0:
        uav_run_img = env.render(return_img = True)
        imageio.mimsave(f"./result/Q_learning/{current_time}_time_index{process_index}_UAV_result.gif",[uav_run_img],fps = 10)
    
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
    ]
    
    for params in parmas_arr:
        for process_index in range(0, num_processes, 1):
            total_data = 0
            agent = None

            env = DeliveryEnvironment(num_points, max_box, process_index)

            # 感測器初始座標 (水下定錨座標)
            init_X = np.array(env.x)
            init_Y = np.array(env.y)

            init_position = [init_X, init_Y]

            env_mutihop = copy.deepcopy(env)
            env_NJNP = copy.deepcopy(env)
            env_greedy_and_mutihop = copy.deepcopy(env)
            env_drift_greedy_and_mutihop = copy.deepcopy(env)
            env_Q = copy.deepcopy(env)
            
            env_NJNP.uav_data_amount_list = copy.deepcopy(env_NJNP.data_amount_list)
            env_greedy_and_mutihop.uav_data_amount_list = copy.deepcopy(env_greedy_and_mutihop.data_amount_list)
            env_drift_greedy_and_mutihop.uav_data_amount_list = copy.deepcopy(env_drift_greedy_and_mutihop.data_amount_list)
            env_Q.uav_data_amount_list = copy.deepcopy(env_Q.data_amount_list)
            
            # NJNP
            parent_num = set_tree_parent_num(env)
            NJNP_nodes = run_NJNP(env, parent_num)
            njnp_nodes = []

            for current_time in range(1, env.run_time + 1, 1):
                
                # 1. mutihop
                # # =============== mutihop ===============
                # uav_run_img = env_mutihop.render(return_img = True)
                # imageio.mimsave(f"./result/mutihop/{index}_loop_index{num+1}_UAV_result.gif",[uav_run_img],fps = 10)
                # # =============== mutihop end ===============

                # 2. greedy
                # =============== env_NJNP ===============
                
                if len(njnp_nodes) == 0:
                    njnp_nodes = copy.deepcopy(NJNP_nodes)
                
                env_NJNP = run_n_NJNP(
                    env_NJNP, 
                    init_position=init_position,
                    current_time=current_time,
                    process_index=process_index,
                    NJNP_nodes=njnp_nodes,
                )
                # =============== env_NJNP end ===============

                # =============== env_greedy_and_mutihop ===============
                env_greedy_and_mutihop = run_n_greedy_mutihop(
                    env_greedy_and_mutihop, 
                    init_position=init_position,
                    current_time=current_time,
                    process_index=process_index,
                )
                # =============== env_greedy_and_mutihop end ===============

                
                # =============== drift greedy and mutihop ===============
                env_drift_greedy_and_mutihop = run_n_greedy_drift(
                    env_drift_greedy_and_mutihop, 
                    init_position=init_position,
                    current_time=current_time,
                    process_index=process_index,
                )

                # =============== drift greedy and mutihop ===============

                # =============== Q learning ===============
                if len(env_Q.stops) == 1:
                    agent = DeliveryQAgent(
                        states_size=num_points,
                        actions_size=num_points,
                        epsilon = 1.0,
                        epsilon_min = params["epsilon_min"],
                        epsilon_decay = 0.9998,
                        gamma = params["gamma"],
                        lr = params["lr"]
                    )

                    # 跑 Q learning
                    env_Q,agent = run_n_episodes(
                        env_Q, 
                        agent,
                        n_episodes=n_episodes,
                        process_index=process_index,
                        current_time=current_time,
                        train_params=params,
                    )
                    
                    
                    route,cost = optimal_route(env.stops, env)
                    startIndex = env.first_point
                    env.stops = route[startIndex:] + route[:startIndex]
                    env.stops.append(env.first_point)
                    
                

                # uav 開始飛行
                env_Q = run_uav(env_Q, init_position, current_time, process_index)
                # =============== Q learning end ===========

                # 減去 mutihop 的資料量 (GPSR)
                if current_time % env.unit_time == 0:    
                    env_NJNP.subtract_mutihop_data()
                    env_greedy_and_mutihop.subtract_mutihop_data()
                    env_drift_greedy_and_mutihop.subtract_mutihop_data()
                    env_Q.subtract_mutihop_data()

                # 執行節點飄移
                env.drift_node(init_position, current_time)
                
                # 將飄移後的節點座標存放到各個環境中
                # env_mutihop.x = np.array(env.x)
                # env_mutihop.y = np.array(env.y)
                env_NJNP.x = np.array(env.x)
                env_NJNP.y = np.array(env.y)
                env_greedy_and_mutihop.x = np.array(env.x)
                env_greedy_and_mutihop.y = np.array(env.y)
                env_drift_greedy_and_mutihop.x = np.array(env.x)
                env_drift_greedy_and_mutihop.y = np.array(env.y)
                env_Q.x = np.array(env.x)
                env_Q.y = np.array(env.y)

                # 產生資料
                if current_time % env.unit_time == 0:
                    # env_mutihop.generate_data(current_time)
                    env_NJNP.generate_data(current_time)
                    env_greedy_and_mutihop.generate_data(current_time)
                    env_drift_greedy_and_mutihop.generate_data(current_time)
                    env_Q.generate_data(current_time)
                    
    print(f'run {index} end ========================================')

# mutiprocessing start ================================
if __name__ == '__main__':
    process_list = []

    for i in range(num_processes):
        process_list.append(mp.Process(target=runMain, args=(i,)))
        process_list[i].start()

    for i in range(num_processes):
        process_list[i].join()

# mutiprocessing end ================================