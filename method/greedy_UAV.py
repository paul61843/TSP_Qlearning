import numpy as np
from tqdm.notebook import tqdm
import math
import time
import imageio
import copy

from utils.calc import *
import csv_utils
from constants.constants import *

# 取得資料量最高的感測器
def getMostDataOfSensor(env):
    max_value = -1
    max_node = None
    

    for i in env.unvisited_stops:
        data_amount = env.uav_data_amount_list[i]['origin'] + env.uav_data_amount_list[i]['calc'] * env.calc_data_compression_ratio
        if data_amount >= max_value:
            max_value = data_amount
            max_node = i

    return max_node

def run_n_greedy(
    env,
    init_position=[],
    current_time=0,
    process_index=0,
):
    start = time.time()

    env.uav_remain_run_distance = env.uav_remain_run_distance + env.uav_speed * 1 # 每秒新增的距離

    # 如果沒有下一個節點，則取得資料量最高的感測器新增
    if len(env.stops) == 1:
        env.unvisited_stops = env.get_unvisited_stops()
        a = getMostDataOfSensor(env)
        env.stops.append(a)

    # 判斷無人機飛是否抵達下一個節點
    x = env.x[[env.stops[-1], env.stops[-2]]]
    y = env.y[[env.stops[-1], env.stops[-2]]]
    next_point_distance = math.ceil(calcDistance(x, y))
    if env.uav_remain_run_distance >= next_point_distance:
        env.uav_remain_run_distance = env.uav_remain_run_distance - next_point_distance
        env.clear_data_one(init_position, env.stops[-1], False)

        # 如果抵達 sink，則 reset 環境
        if env.stops[-1] == env.first_point:
            env.stops = []
            env.stops.append(env.first_point)
            env.uav_data_amount_list = copy.deepcopy(env.data_amount_list)

        # 新增下一個節點
        env.unvisited_stops = env.get_unvisited_stops()
        a = getMostDataOfSensor(env)
        env.stops.append(a)

        # 判斷無人機飛行距離是否能返回 sink
        x = env.x[[env.stops[0], env.stops[-1]]]
        y = env.y[[env.stops[0], env.stops[-1]]]
        to_start_cost = calcDistance(x, y)
        distance = calcDistance(env.x[env.stops], env.y[env.stops]) + to_start_cost
        if distance > env.max_move_distance:
            env.stops.pop()
            env.stops.append(env.first_point)

    # 紀錄資料
    if current_time % env.unit_time == 0:
        mutihop_data = env.sum_mutihop_data * env.calc_data_compression_ratio
        sensor_data_origin = sum(item['origin'] for item in env.data_amount_list)
        sensor_data_calc = sum(item['calc'] for item in env.data_amount_list) * env.calc_data_compression_ratio
        sensor_data = sensor_data_origin + sensor_data_calc
        
        uav_data = env.uav_data['origin'] + env.uav_data['calc'] * env.calc_data_compression_ratio
        
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
            math.ceil(uav_data // 8), 
            math.ceil(lost_data // 8),
        ])
        csv_utils.writeDataToCSV(f'./result/csv{process_index}/greedy.csv', env.result)

    # 產生UAV路徑圖
    if current_time % 1000 == 0:
        uav_run_img = env.render(return_img = True)
        imageio.mimsave(f"./result/greedy/{current_time}_time_index{process_index}_UAV_result.gif",[uav_run_img],fps = 10)
    
    end = time.time()
    
    return env

def run_n_greedy_mutihop(
    env,
    init_position=[],
    current_time=0,
    process_index=0,
):
    start = time.time()

    env.uav_remain_run_distance = env.uav_remain_run_distance + env.uav_speed * 1 # 每秒新增的距離

    # 如果沒有下一個節點，則取得資料量最高的感測器新增
    if len(env.stops) == 1:
        env.unvisited_stops = env.get_unvisited_stops()
        a = getMostDataOfSensor(env)
        env.stops.append(a)

    # 判斷無人機飛是否抵達下一個節點
    x = env.x[[env.stops[-1], env.stops[-2]]]
    y = env.y[[env.stops[-1], env.stops[-2]]]
    next_point_distance = math.ceil(calcDistance(x, y))
    if env.uav_remain_run_distance >= next_point_distance:
        env.uav_remain_run_distance = env.uav_remain_run_distance - next_point_distance
        env.clear_data_one(init_position, env.stops[-1], False)

        # 如果抵達 sink，則 reset 環境
        if env.stops[-1] == env.first_point:
            print(env)
            env.stops = []
            env.stops.append(env.first_point)
            env.uav_data_amount_list = copy.deepcopy(env.data_amount_list)

        # 新增下一個節點
        env.unvisited_stops = env.get_unvisited_stops()
        a = getMostDataOfSensor(env)
        env.stops.append(a)

        # 判斷無人機飛行距離是否能返回 sink
        x = env.x[[env.stops[0], env.stops[-1]]]
        y = env.y[[env.stops[0], env.stops[-1]]]
        to_start_cost = calcDistance(x, y)
        distance = calcDistance(env.x[env.stops], env.y[env.stops]) + to_start_cost
        if distance > env.max_move_distance:
            env.stops.pop()
            env.stops.append(env.first_point)

    # 紀錄資料
    if current_time % env.unit_time == 0:
        mutihop_data = env.sum_mutihop_data * env.calc_data_compression_ratio
        sensor_data_origin = sum(item['origin'] for item in env.data_amount_list)
        sensor_data_calc = sum(item['calc'] for item in env.data_amount_list) * env.calc_data_compression_ratio
        sensor_data = sensor_data_origin + sensor_data_calc
        
        uav_data = env.uav_data['origin'] + env.uav_data['calc'] * env.calc_data_compression_ratio
        
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
            math.ceil(uav_data // 8), 
            math.ceil(lost_data // 8),
        ])
        csv_utils.writeDataToCSV(f'./result/csv{process_index}/greedy_and_mutihop.csv', env.result)

    # 產生UAV路徑圖
    if current_time % 1000 == 0:
        uav_run_img = env.render(return_img = True)
        imageio.mimsave(f"./result/greedy_and_mutihop/{current_time}_time_index{process_index}_UAV_result.gif",[uav_run_img],fps = 10)
    
    end = time.time()
    
    return env


def run_n_greedy_drift(
    env,
    init_position=[],
    current_time=0,
    process_index=0,
):
    start = time.time()

    env.uav_remain_run_distance = env.uav_remain_run_distance + env.uav_speed * 1 # 每秒新增的距離

    # 如果沒有下一個節點，則取得資料量最高的感測器新增
    if len(env.stops) == 1:
        env.unvisited_stops = env.get_unvisited_stops()
        a = getMostDataOfSensor(env)
        env.stops.append(a)

    # 判斷無人機飛是否抵達下一個節點
    x = env.x[[env.stops[-1], env.stops[-2]]]
    y = env.y[[env.stops[-1], env.stops[-2]]]
    next_point_distance = math.ceil(calcDistance(x, y))
    if env.uav_remain_run_distance >= next_point_distance:
        env.uav_remain_run_distance = env.uav_remain_run_distance - next_point_distance
        env.clear_data_one(init_position, env.stops[-1], True)

        # 如果抵達 sink，則 reset 環境
        if env.stops[-1] == env.first_point:
            env.stops = []
            env.stops.append(env.first_point)
            env.uav_data_amount_list = copy.deepcopy(env.data_amount_list)

        # 新增下一個節點
        env.unvisited_stops = env.get_unvisited_stops()
        a = getMostDataOfSensor(env)
        env.stops.append(a)

        # 判斷無人機飛行距離是否能返回 sink
        x = env.x[[env.stops[0], env.stops[-1]]]
        y = env.y[[env.stops[0], env.stops[-1]]]
        to_start_cost = calcDistance(x, y)
        distance = calcDistance(env.x[env.stops], env.y[env.stops]) + to_start_cost
        if distance > env.max_move_distance:
            env.stops.pop()
            env.stops.append(env.first_point)

    # 紀錄資料
    if current_time % env.unit_time == 0:
        mutihop_data = env.sum_mutihop_data * env.calc_data_compression_ratio
        sensor_data_origin = sum(item['origin'] for item in env.data_amount_list)
        sensor_data_calc = sum(item['calc'] for item in env.data_amount_list) * env.calc_data_compression_ratio
        sensor_data = sensor_data_origin + sensor_data_calc
        
        uav_data = env.uav_data['origin'] + env.uav_data['calc'] * env.calc_data_compression_ratio
        
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
            math.ceil(uav_data // 8), 
            math.ceil(lost_data // 8),
        ])
        csv_utils.writeDataToCSV(f'./result/csv{process_index}/drift_greedy_and_mutihop.csv', env.result)

    # 產生UAV路徑圖
    if current_time % 1000 == 0:
        uav_run_img = env.render(return_img = True)
        imageio.mimsave(f"./result/drift_greedy_and_mutihop/{current_time}_time_index{process_index}_UAV_result.gif",[uav_run_img],fps = 10)
    
    end = time.time()
    
    return env