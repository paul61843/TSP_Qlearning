import random as rd
from matplotlib import pyplot as plt

from collections import OrderedDict
import math
from utils.calc import *
import csv_utils
import imageio
import copy
import subprocess


def GA_TSP(env):
    result = ''
    
    arr_index = copy.deepcopy(list(set(env.unconnect_nodes + env.buffer_overflow_nodes + [env.first_point])))
    
    x = ' '.join(str(e) for e in env.x[arr_index])
    y = ' '.join(str(e) for e in env.y[arr_index])
    arr_index = ' '.join(str(e) for e in arr_index)

    command = f'node method/TSP.js "{x}" "{y}" "{arr_index}"'
    nodejs = subprocess.Popen(command, stdout=subprocess.PIPE)
    result = nodejs.stdout.read().decode("utf-8").replace('\n', '')


        
    result = [ int(x) for x in result.split()]
        
    first_index = result.index(env.first_point)
    
    result = result[:first_index] + result[first_index:]
    
    return result

def run_TSP(env, init_position, current_time, process_index):
    
    env.uav_remain_run_distance = env.uav_remain_run_distance + env.uav_speed * 1 # 每秒新增的距離
    
    # 判斷無人機飛是否抵達下一個節點
    x = env.x[[env.stops[env.current_run_index], env.stops[env.next_point]]]
    y = env.y[[env.stops[env.current_run_index], env.stops[env.next_point]]]
    
    init_pos_x, init_pos_y = init_position
    
    distance = math.ceil(calcDistance(x, y))
    


    if env.uav_remain_run_distance >= distance:
        
        next_point_cost = distance
        
        if env.uav_remain_run_distance >= next_point_cost:
            env.uav_remain_run_distance = env.uav_remain_run_distance - next_point_cost
            env.clear_data_one(init_position, env.stops[env.next_point], False)
        
            # 如果抵達 sink，則 reset 環境
            if env.stops[env.next_point] == env.stops[-1]:
                env.stops = []
                env.stops.append(env.first_point)
                env.current_run_index = 0
                env.next_point = 1
                env.collected_sensors = []

                return env
            
            # 新增下一個節點
            env.current_run_index = env.current_run_index + 1
            env.next_point = env.next_point + 1
            
        
    # 紀錄資料
    if current_time % env.record_time == 0:
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
        connect_num = len(env.connect_nodes)
        buffer_overflow_num = len(env.buffer_overflow_nodes)
        
        
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
            math.ceil(connect_num),
            math.ceil(buffer_overflow_num),
        ])
        csv_utils.writeDataToCSV(f'./result/csv/csv{process_index}/TSP.csv', env.result)

    # 產生UAV路徑圖
    if current_time % 500 == 0:
        uav_run_img = env.render(return_img = True)
        imageio.mimsave(f"./result/TSP/{current_time}_time_index{process_index}_UAV_result.gif",[uav_run_img],fps = 10)
    
    return env
