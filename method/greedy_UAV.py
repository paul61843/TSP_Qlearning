import numpy as np
from tqdm.notebook import tqdm

from utils.calc import *
import csv_utils
from constants import *

# 取得資料量最高的感測器
def getMostDataOfSensor(data_amount_list, unvisited_stops):
    max_value = -1
    max_node = None

    for i in unvisited_stops:
        data_amount = data_amount_list[i]
        if data_amount >= max_value:
            max_value = data_amount
            max_node = i

    return max_node

def run_n_greedy(
    env,
    name="training.gif",
    n_episodes=1000,
    render_each=10,
    fps=10,
    result_index=0,
    loop_index=0,
    train_params={},
    init_position=[],
):
    # reset stops
    env.stops = []
    env.stops.append(env.first_point)

    for i in env.x:
        env.unvisited_stops = env.get_unvisited_stops()
        a = getMostDataOfSensor(env.data_amount_list, env.unvisited_stops)

        if a == None:
            break
        
        env.stops.append(a)

        distance = calcDistance(env.x[env.stops], env.y[env.stops])

        x = env.x[[env.stops[0], env.stops[-1]]]
        y = env.y[[env.stops[0], env.stops[-1]]]
        to_start_cost = calcDistance(x, y)

        distance = distance + to_start_cost

        if distance > env.max_move_distance:
            env.stops.pop()
            break

    return env

def run_n_greedy_drift(
    env,
    name="training.gif",
    n_episodes=1000,
    render_each=10,
    fps=10,
    result_index=0,
    loop_index=0,
    train_params={},
    init_position=[],
    total_data=0,
):
    # reset stops
    env.stops = []
    env.stops.append(env.first_point)
    
    recordIndex = int(env.current_time // env.unit_time) + 1


    for i in env.x:
        
        env.unvisited_stops = env.get_unvisited_stops()
        a = getMostDataOfSensor(env.data_amount_list, env.unvisited_stops)
        if a == None:
            break

        env.stops.append(a)

        init_pos_x, init_pos_y = init_position
        drift_cost = calc_drift_cost(
            [init_pos_x[a],  env.x[a]], 
            [init_pos_y[a],  env.y[a]], 
            env
        )

        distance = calcDistance(env.x[env.stops], env.y[env.stops])
        
        env.current_time = env.current_time + distance + drift_cost
        
        x = env.x[[env.stops[0], env.stops[-1]]]
        y = env.y[[env.stops[0], env.stops[-1]]]
        to_start_cost = calcDistance(x, y)

        total_cost = distance + to_start_cost + drift_cost

        if total_cost > env.max_move_distance:
            env.stops.pop()
            distance = calcDistance(env.x[env.stops], env.y[env.stops])
            env.current_time = env.current_time + distance + drift_cost
            
            if recordIndex <= int(env.current_time // env.unit_time):
                for i in range(int(env.current_time // env.unit_time) - recordIndex + 1):
                    env.clear_data(init_position, False)
                    env.subtract_mutihop_data()
                    mutihop_data = env.sum_mutihop_data
                    sensor_data = sum(env.data_amount_list)
                    total_data = env.generate_data_total

                    lost_data = env.generate_data_total - (mutihop_data + sensor_data + env.uav_data)
                    run_time = recordIndex * env.unit_time
                    env.result.append([run_time, total_data, mutihop_data, sensor_data, env.uav_data, lost_data])
                    
                    recordIndex = recordIndex + 1
                    added_data = generate_data_50[recordIndex % len(generate_data_50)]
                    env.generate_data_total = env.generate_data_total + sum(added_data)
                    env.generate_data(added_data)
                    
            break
        
        if recordIndex <= int(env.current_time // env.unit_time):
            for i in range(int(env.current_time // env.unit_time) - recordIndex + 1):
                env.clear_data(init_position, False)
                env.subtract_mutihop_data()
                mutihop_data = env.sum_mutihop_data
                sensor_data = sum(env.data_amount_list)
                total_data = env.generate_data_total
                lost_data = total_data - (mutihop_data + sensor_data + env.uav_data)
                run_time = recordIndex * env.unit_time
                env.result.append([run_time, total_data, mutihop_data, sensor_data, env.uav_data, lost_data])
                
                recordIndex = recordIndex + 1
                added_data = generate_data_50[recordIndex % len(generate_data_50)]
                env.generate_data_total = env.generate_data_total + sum(added_data)
                env.generate_data(added_data)


    return env