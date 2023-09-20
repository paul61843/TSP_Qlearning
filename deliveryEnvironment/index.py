import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from utils.calc import *
from isolated_nodes.index import *
from constants import *


class DeliveryEnvironment(object):
    def __init__(self,n_stops = 20,max_box = 10,method = "distance",**kwargs):

        print(f"Initialized Delivery Environment with {n_stops} random stops")
        print(f"Target metric for optimization is {method}")

        # Environment Config
        self.point_range = 10 # 節點通訊半徑範圍 (單位 10m)
        self.drift_range = 6 # 節點飄移範圍 (單位 10m)
        self.system_time = 5000 # 執行時間 (單位s)
        self.unit_time = 100 # 時間單位 (單位s)
        self.current_time = 0 # 目前時間 (單位s)
        
        # UAV Config
        self.uav_range = 5 # 無人機通訊半徑範圍 (單位 10m)
        self.uav_speed = 0.5 # 無人機移動速度 (單位 10m/s)
        self.uav_energy = 100 * 1000 # 無人機電量 (單位w)
        self.uav_energy_consumption = 200 # 無人機每秒消耗電量 (單位w)
        self.uav_flyTime = self.uav_energy / self.uav_energy_consumption # 無人機可飛行時間 (單位s)
        self.max_move_distance = 500 # 無人機最大移動距離 (單位 10m)
        

        # 無人機探索，飄移節點最大能量消耗
        # 假設無人機只需飛行一圈，即可完整探索感測器飄移可能區域
        # 故無人機只需以 r/2 為半徑飛行
        self.drift_max_cost = 2 * (self.drift_range / 2) * math.pi  # 公式 2 x 3.14 x (r/2)

        self.data_generatation_range = 20 # 節點資料從 1 ~ 20 數字中隨機增加
        self.mutihop_transmission = 1 # 透過muti-hop方式 減少的資料量


        # Initialization
        self.n_stops = n_stops
        self.action_space = self.n_stops
        self.observation_space = self.n_stops
        self.remain_power = self.max_move_distance
        self.max_box = max_box
        self.stops = []
        self.unvisited_stops = []
        self.red_stops = []
        self.drift_cost_list = []
        self.method = method
        self.result = []

        # 資料
        self.uav_data = 0
        self.sum_mutihop_data = 0
        self.generate_data_total = 0

        # 感測器資料量相關
        self.data_amount_list = [] # 感測器儲存的資料量
        self.buffer_size = 50 # 感測器儲存資料的最大量
        self.calc_threshold = self.buffer_size * 50 # 感測器資料量超過 50% 門檻
        self.calc_danger_threshold = self.buffer_size * 75 # 感測器資料量超過 75% 門檻

        # 隔離節點
        self.isolated_node = []

        # Generate stops
        self._generate_stops()
        self._generate_q_values()
        self.render()

        # Initialize first point
        self.set_first_point()
        self.reset()

    def _generate_stops(self):
        use_fake_data = True

        points = np.random.rand(1,2) * self.max_box

        # 隨機生成感測器數量，並確保每個點的通訊範圍內至少有一個點
        if use_fake_data:
            self.x = xPoints 
            self.y = yPoints
        else:
            while (len(points) < self.n_stops):
                x,y = (np.random.rand(1,2) * self.max_box)[0]
                for p in points:
                    isTrue = any(
                        (((x - p[0]) ** 2 + (y - p[1]) ** 2 ) ** 0.5 <= self.point_range) and 
                        (((x - p[0]) ** 2 + (y - p[1]) ** 2 ) ** 0.5 >= self.point_range / 2) 
                        for p in points
                    )
                    if isTrue:
                        points = np.append(points, [np.array([x,y])], axis=0)
                        break
            self.x = points[:,0]
            self.y = points[:,1]

        # 預設感測器的目前資料量為0
        self.data_amount_list = [0] * self.n_stops
        
    def set_isolated_node(self):
        points = []
        
        for idx, i in enumerate(self.x):
            points.append((self.x[idx], self.y[idx]))
        first_point = calcNodeToOriginDistance(self)

        self.isolated_node = find_isolated_nodes(
            points, 
            self.point_range, 
            (self.x[first_point], self.y[first_point])
        )

        
    # 加上隨機產生感測器的資料量 max = 100
    def generate_data(self, add_data):
        arr1 = self.data_amount_list
        arr2 = add_data
        
        added_data = [ x + y for x, y in zip(arr1, arr2) ]
        self.data_amount_list = [ x if x <= self.buffer_size else self.buffer_size for x in added_data ]

    # 減去 muti hop 傳輸的資料
    def subtract_mutihop_data(self):
        arr = np.array(self.data_amount_list)

        for i, data in enumerate(arr):
            not_isolated_node = i not in self.isolated_node
            if not_isolated_node:
                self.sum_mutihop_data = self.sum_mutihop_data + (arr[i] - max(arr[i] - self.mutihop_transmission, 0))
                arr[i] = max(arr[i] - self.mutihop_transmission, 0)

        self.data_amount_list = arr

    def get_unvisited_stops(self):
        # 使用 set 運算來找出未被包含在 route 中的節點
        unvisited_stops = set(list(range(0, self.n_stops))) - set(self.stops)
        return list(unvisited_stops)
    
    # 清除無人跡拜訪後的感測器資料
    def clear_data(self, init_position, drift_consider):
        [init_x, init_y] = init_position
        for i in self.stops:
            drift_distance = np.sqrt((self.x[i] - init_x[i]) ** 2 + (self.y[i] - init_y[i]) ** 2)
            if (drift_distance <= self.uav_range) or drift_consider:
                self.uav_data = self.uav_data + self.data_amount_list[i]
                self.data_amount_list[i] = 0

    def _generate_q_values(self,box_size = 0.2):
        xy = np.column_stack([self.x,self.y])
        self.q_stops = cdist(xy,xy)


    def render(self,return_img = False):
        
        fig = plt.figure(figsize=(7,7))
        ax = plt.axes()
        plt.title("Delivery Stops")
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        
        # Show stops
        plt.scatter(self.x,self.y,c = "black",s = 30)
        
        self.red_stops = []

        # 將孤立節點標記為灰色
        for i in self.isolated_node:
            plt.scatter(self.x[i], self.y[i], c = "#AAAAAA", s = 30)

        # 感測器的資料量大於50，將節點標記為黃色、紅色(代表優先節點)
        for i in range(self.n_stops):
            if self.data_amount_list[i] > self.calc_threshold:
                plt.scatter(self.x[i], self.y[i], c = "yellow", s = 30)

            if self.data_amount_list[i] > self.calc_danger_threshold:
                self.red_stops.append(i)
                plt.scatter(self.x[i], self.y[i], c = "red", s = 30) 

        # Show START
        if len(self.stops) > 0:
            xy = self._get_xy(initial = True)
            xytext = xy[0] + 0.1, xy[1]-0.05
            plt.annotate("SINK",xy=xy,xytext=xytext,weight = "bold")
            
        # Show itinerary
        if len(self.stops) > 1:
            x = np.concatenate((self.x[self.stops], [self.x[self.stops[0]]]))
            y = np.concatenate((self.y[self.stops], [self.y[self.stops[0]]]))
            plt.plot(x, y, c = "blue",linewidth=1,linestyle="--")
            
        plt.xticks(list(range(0, self.max_box + 20, 10)))
        plt.yticks(list(range(0, self.max_box + 20, 10)))
        
        if return_img:
            # From https://ndres.me/post/matplotlib-animated-gifs-easily/
            fig.canvas.draw()
            # fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close('all')

            return image
        else:
            print('render')
            # plt.show()



    def reset(self):

        # Stops placeholder
        self.stops = []
        self.drift_cost_list = len(self.stops) * [self.drift_max_cost]
        self.stops.append(self.first_point)


    def set_first_point(self):
        self.first_point = calcNodeToOriginDistance(self)

    def step(self,destination):

        # Get current state
        state = self._get_state()
        new_state = destination

        # Get reward for such a move
        reward = self._get_reward(state,new_state)

        # Append new_state to stops
        self.stops.append(destination)
        done = len(self.stops) == self.n_stops

        return new_state,reward,done

    def drift_node(self):
        for i in range(1, len(self.x)):
            if i != self.first_point:
                self.x[i] = self.x[i] + random.uniform(-self.drift_range, self.drift_range)
                self.x[i] = 0 if self.x[i] <= 0 else self.max_box if self.x[i] >= self.max_box else self.x[i]
                
                self.y[i] = self.y[i] + random.uniform(-self.drift_range, self.drift_range)
                self.y[i] = 0 if self.y[i] <= 0 else self.max_box if self.y[i] >= self.max_box else self.y[i]
                
        

    def _get_state(self):
        return self.stops[-1]


    def _get_xy(self,initial = False):
        state = self.first_point if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x,y

    def _get_reward(self,state,new_state):
        trade_of_factor = 0.001

        distance = self.q_stops[state,new_state]
        distance_reward = (1 - trade_of_factor * distance ** 2)

        has_calc_threshold = self.data_amount_list[new_state] > self.calc_threshold
        has_calc_danger_threshold = self.data_amount_list[new_state] > self.calc_danger_threshold

        yellow_reward = has_calc_threshold * 1
        danger_reward = has_calc_danger_threshold * 6
        

        # 新增 孤立節點獎勵值

        is_isolated_node = new_state in self.isolated_node
        isolated_reward = 2 if is_isolated_node else 0

        # unvisited_stops = self.get_unvisited_stops()
        # calcAvg(new_state, unvisited_stops, self)

        return distance_reward + yellow_reward + danger_reward + isolated_reward
        # return (1 - trade_of_factor * distance ** 2)
        # return -(distance ** 2)
        # return -distance
        # return Theta_one * distance + 
        #        (1 - Theta_one) * Theta_two / 

