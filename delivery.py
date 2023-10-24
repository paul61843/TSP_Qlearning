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
from method.method1 import *
from method.method2 import *


plt.style.use("seaborn-v0_8-dark")

sys.path.append("../")

# 待完成事項

# 無人機 不具備飄移探索能力


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

# 找現成的感測器 buffer 大小

# 1k 240 秒
# 16k



# 設定環境參數
num_processes = 1 # 同時執行數量 (產生結果數量)
num_points = 400 # 節點數
max_box = 2000  # 場景大小 單位 (1m)

n_episodes = 5000 # 訓練次數

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



def run_uav(env, init_position):

    idx = 0
    
    recordIndex = int(env.current_time // env.unit_time)

    total_drift_cost = 0

    current_time = 0

    env.uav_data_amount_list = env.data_amount_list

    temp_points = []

    while idx < len(env.stops):


        route = env.stops[idx]

        init_pos_x, init_pos_y = init_position

        position_x = env.x[env.stops]
        position_y = env.y[env.stops]

        temp_points.append(route)

        # 搜尋飄移節點 所需要的電量消耗
        drift_cost = calc_drift_cost(
            [init_pos_x[route],  position_x[idx]], 
            [init_pos_y[route],  position_y[idx]], 
            env
        )

        env.drift_cost_list[idx] = drift_cost

        total_drift_cost = total_drift_cost + drift_cost

        drift_remain_cost = env.drift_max_cost - drift_cost
        # 用於搜尋飄移節點中的剩餘能量
        env.remain_power = env.remain_power + drift_remain_cost
        env.unvisited_stops = env.get_unvisited_stops()
        mostDataOfPoint = getMostDataOfSensor(env.uav_data_amount_list, env.unvisited_stops)

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
                env.remain_power = new_cost

        idx = idx + 1

        distance = calcDistance(env.x[env.stops[:idx]], env.y[env.stops[:idx]]) + total_drift_cost
        current_time = env.current_time + distance // env.uav_speed

        if recordIndex <= int(current_time // env.unit_time):

            for i in temp_points:
                env.clear_data_one(init_position, i , False)
                temp_points = []

            for i in range(int(current_time // env.unit_time) - recordIndex):
                # env.clear_data(init_position, False)
                env.subtract_mutihop_data()
                    
                mutihop_data = env.sum_mutihop_data
                sensor_data = sum(env.data_amount_list)
                total_data = env.generate_data_total
                lost_data = total_data - (mutihop_data + sensor_data + env.uav_data)

                run_time = recordIndex * env.unit_time
                env.result.append([
                    math.ceil(run_time), 
                    math.ceil(total_data),
                    math.ceil(mutihop_data), 
                    math.ceil(sensor_data), 
                    math.ceil(env.uav_data), 
                    math.ceil(lost_data),
                ])
                
                added_event_data = generate_data_50[recordIndex % len(generate_data_50)][:env.n_stops]
                added_min_data = env.min_generate_data * len(added_event_data)
                env.generate_data_total = env.generate_data_total + sum(added_event_data) + added_min_data
                env.generate_data(added_event_data)

                recordIndex = recordIndex + 1

    
    distance = calcDistance(env.x[env.stops], env.y[env.stops]) + sum(env.drift_cost_list)
    env.current_time = env.current_time + distance // env.uav_speed

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
    ]
    
    for params in parmas_arr:
        for field_index in range(0, num_processes, 1):
        # for field_index in range(0, len(sensor_position), 2):
            total_data = 0

            env = DeliveryEnvironment(num_points, max_box, field_index)

            env_mutihop = copy.deepcopy(env)
            env_greedy = copy.deepcopy(env)
            env_greedy_and_mutihop = copy.deepcopy(env)
            env_drift_greedy_and_mutihop = copy.deepcopy(env)
            env_Q = copy.deepcopy(env)

            # 感測器初始座標 (水下定錨座標)
            init_X = np.array(env.x)
            init_Y = np.array(env.y)
            init_position = [init_X, init_Y]

            num_uav_loops = int(env.system_time // env.uav_flyTime)
                
            for num in range(num_uav_loops):
                env_mutihop.x = np.array(env.x)
                env_mutihop.y = np.array(env.y)
                env_greedy.x = np.array(env.x)
                env_greedy.y = np.array(env.y)
                env_greedy_and_mutihop.x = np.array(env.x)
                env_greedy_and_mutihop.y = np.array(env.y)
                env_drift_greedy_and_mutihop.x = np.array(env.x)
                env_drift_greedy_and_mutihop.y = np.array(env.y)
                env_Q.x = np.array(env.x)
                env_Q.y = np.array(env.y)

                # 判斷是否為孤立節點
                env.set_isolated_node()
                env_mutihop.set_isolated_node()
                env_greedy.set_isolated_node()
                env_greedy_and_mutihop.set_isolated_node()
                env_drift_greedy_and_mutihop.set_isolated_node()
                env_Q.set_isolated_node()

                # =============== Q learning ===============
                print('Q learning start')
                start = time.time()

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
                    result_index=index,
                    loop_index=num+1,
                    train_params=params,
                )

                # uav 開始飛行
                env_Q = run_uav(env_Q, init_position)

                # 產生UAV路徑圖
                uav_run_img = env_Q.render(return_img = True)
                imageio.mimsave(f"./result/Q_learning/{index}_epsilon_min{params['epsilon_min']}_gamma{params['gamma']}_lr{params['lr']}_loop_index{num+1}_UAV_result.gif",[uav_run_img],fps = 10)
                csv_utils.writeDataToCSV(f'./result/csv{index}/q_learning.csv', env_Q.result)
                
                end = time.time()
                print('Q learning end', end - start)
                # =============== Q learning end ===========
                if env.stops == []:
                    print('no stops')
                    break

                # =============== mutihop ===============
                uav_run_img = env_mutihop.render(return_img = True)
                imageio.mimsave(f"./result/mutihop/{index}_loop_index{num+1}_UAV_result.gif",[uav_run_img],fps = 10)
                # =============== mutihop end ===============

                # =============== env_greedy ===============
                print('env_greedy start')
                start = time.time()

                # 跑 uav greedy
                env_greedy = run_n_greedy(
                    env_greedy, 
                    n_episodes=n_episodes,
                    result_index=index,
                    loop_index=num+1,
                    train_params=params,
                    init_position=init_position,
                    total_data=total_data,
                )

                # 產生UAV路徑圖
                uav_run_img = env_greedy.render(return_img = True)
                imageio.mimsave(f"./result/greedy/{index}_loop_index{num+1}_UAV_result.gif",[uav_run_img],fps = 10)
                csv_utils.writeDataToCSV(f'./result/csv{field_index}/greedy.csv', env_greedy.result)
                end = time.time()
                
                print('env_greedy end', end - start)

                # =============== env_greedy end ===============


                # =============== greedy and mutihop ===========
                print('greedy and mutihop start')
                start = time.time()


                env_greedy_and_mutihop = run_n_greedy_mutihop(
                    env_greedy_and_mutihop, 
                    n_episodes=n_episodes,
                    result_index=index,
                    loop_index=num+1,
                    train_params=params,
                    init_position=init_position,
                    total_data=total_data,
                )

                # 產生UAV路徑圖
                uav_run_img = env_greedy_and_mutihop.render(return_img = True)
                imageio.mimsave(f"./result/greedy_and_mutihop/{index}_loop_index{num+1}_UAV_result.gif",[uav_run_img],fps = 10)
                csv_utils.writeDataToCSV(f'./result/csv{field_index}/greedy_and_mutihop.csv', env_greedy_and_mutihop.result)
                
                end = time.time()
                print('greedy and mutihop end', end - start)
                # =============== greedy and mutihop ===============

                # =============== drift greedy and mutihop ===============
                print('drift greedy and mutihop start')
                start = time.time()

                # 跑 uav greedy
                env_drift_greedy_and_mutihop = run_n_greedy_drift(
                    env_drift_greedy_and_mutihop, 
                    n_episodes=n_episodes,
                    result_index=index,
                    loop_index=num+1,
                    train_params=params,
                    init_position=init_position,
                    total_data=total_data,
                )

                # 產生UAV路徑圖
                uav_run_img = env_drift_greedy_and_mutihop.render(return_img = True)
                imageio.mimsave(f"./result/drift_greedy_and_mutihop/{index}_loop_index{num+1}_UAV_result.gif",[uav_run_img],fps = 10)
                csv_utils.writeDataToCSV(f'./result/csv{field_index}/drift_greedy_and_mutihop.csv', env_drift_greedy_and_mutihop.result)
                

                end = time.time()
                print('drift greedy and mutihop end', end - start)

                # =============== drift greedy and mutihop ===============

                # 清除無人跡拜訪後的感測器資料
                # env_greedy.clear_data(init_position, False)
                # env_greedy_and_mutihop.clear_data(init_position, False)
                # env_drift_greedy_and_mutihop.clear_data(init_position, True)
                # env_Q.clear_data(init_position, True)


                # 感測器儲存的資料，減去mutihop幫傳的資料
                # env_mutihop.subtract_mutihop_data()
                # env_greedy_and_mutihop.subtract_mutihop_data()
                # env_drift_greedy_and_mutihop.subtract_mutihop_data()
                # env_Q.subtract_mutihop_data()

                g_mutihop_data = env_greedy.sum_mutihop_data
                gam_mutihop_data = env_greedy_and_mutihop.sum_mutihop_data
                dgam_mutihop_data = env_drift_greedy_and_mutihop.sum_mutihop_data
                q_mutihop_data = env_Q.sum_mutihop_data
                m_mutihop_data = env_mutihop.sum_mutihop_data

                g_sensor_data = sum(env_greedy.data_amount_list)
                gam_sensor_data = sum(env_greedy_and_mutihop.data_amount_list)
                dgam_sensor_data = sum(env_drift_greedy_and_mutihop.data_amount_list)
                q_sensor_data = sum(env_Q.data_amount_list)
                m_sensor_data = sum(env_mutihop.data_amount_list)

                g_sensor_lost = total_data - (g_sensor_data + env_greedy.uav_data)
                gam_sensor_lost = total_data - (gam_mutihop_data + gam_sensor_data + env_greedy_and_mutihop.uav_data)
                dgam_sensor_lost = total_data - (dgam_mutihop_data + dgam_sensor_data + env_drift_greedy_and_mutihop.uav_data)
                q_sensor_lost = total_data - (q_mutihop_data + q_sensor_data + env_Q.uav_data)
                m_sensor_lost = total_data - (m_mutihop_data + m_sensor_data + env_mutihop.uav_data)

                csv_utils.write('./result/train_table.csv', 
                    [
                        # 系統產生的總資料量、感測器內的資料量、UAV幫助的資料量
                        [''.ljust(30),                        'total_data'.ljust(15)    ,  'mutihop'.ljust(15)               ,  'sensor_data'.ljust(15)                 ,   'uav_data'.ljust(15),                                 'lost_data'.ljust(15),             'sum'.ljust(15)                                                                               ],
                        ['greedy'.ljust(30) ,                  str(total_data).ljust(15),  str(g_mutihop_data).ljust(15)     ,   str(g_sensor_data).ljust(15)           ,   str(env_greedy.uav_data).ljust(15),                   str(g_sensor_lost).ljust(15),      str(g_sensor_data + env_greedy.uav_data).ljust(15)                                           ],
                        ['mutihop'.ljust(30),                  str(total_data).ljust(15),  str(m_mutihop_data).ljust(15)     ,   str(m_sensor_data).ljust(15)           ,   str(env_mutihop.uav_data).ljust(15),                  str(m_sensor_lost).ljust(15),      str(m_mutihop_data + m_sensor_data + env_mutihop.uav_data).ljust(15)                         ],
                        ['greedy_and_mutihop'.ljust(30),       str(total_data).ljust(15),  str(gam_mutihop_data).ljust(15)   ,   str(gam_sensor_data).ljust(15)         ,   str(env_greedy_and_mutihop.uav_data).ljust(15),       str(gam_sensor_lost).ljust(15),    str(gam_mutihop_data + gam_sensor_data + env_greedy_and_mutihop.uav_data).ljust(15)          ],
                        ['drift_greedy_and_mutihop'.ljust(30), str(total_data).ljust(15),  str(dgam_mutihop_data).ljust(15)  ,   str(dgam_sensor_data).ljust(15)        ,   str(env_drift_greedy_and_mutihop.uav_data).ljust(15), str(dgam_sensor_lost).ljust(15),   str(dgam_mutihop_data + dgam_sensor_data + env_drift_greedy_and_mutihop.uav_data).ljust(15)  ],
                        ['Q_learning'.ljust(30),               str(total_data).ljust(15),  str(q_mutihop_data).ljust(15)     ,   str(q_sensor_data).ljust(15)           ,   str(env_Q.uav_data).ljust(15),                        str(q_sensor_lost).ljust(15),      str(q_mutihop_data + q_sensor_data + env_Q.uav_data).ljust(15)                               ],
                    ]
                )

                # 產生場景的方式
                # if len(env.isolated_node) <= 10:
                #     env.generate_stops_and_remove_drift_point()

                env.x = np.array(init_X)
                env.y = np.array(init_Y)

                # 執行節點飄移
                env.drift_node(num)

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