import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from utils.calc import *


class DeliveryEnvironment(object):
    def __init__(self,n_stops = 20,max_box = 10,method = "distance",**kwargs):

        print(f"Initialized Delivery Environment with {n_stops} random stops")
        print(f"Target metric for optimization is {method}")

        # Environment Config
        self.point_range = 20 # 節點通訊範圍 (單位m)
        self.max_move_distance = 300 # 無人機最大移動距離 (單位m)

        self.drift_range = 1 # 節點飄移範圍
        self.drift_max_cost = 2 * self.drift_range * math.pi # 無人機探索，飄移節點最大能量消耗

        self.data_generatation_range = 20 # 節點產生的資料量範圍


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

        # 感測器資料量相關
        self.calc_amount = [] # 感測器儲存的資料量
        self.calc_threshold = 50
        self.calc_danger_threshold = 75

        # 隔離節點
        self.isolated_node = []

        # Generate stops
        self._generate_stops()
        self._generate_q_values()
        self.set_isolated_node()
        self.render()

        # Initialize first point
        self.reset()

    def _generate_stops(self):
        points = np.random.rand(1,2) * self.max_box

        # 隨機生成感測器數量，並確保每個點的通訊範圍內至少有一個點
        while (len(points) < self.n_stops):
            x,y = (np.random.rand(1,2) * self.max_box)[0]
            for p in points:
                isTrue = any(((x - p[0]) ** 2 + (y - p[1]) ** 2 ) ** 0.5 <= self.point_range for p in points)
                if isTrue:
                    points = np.append(points, [np.array([x,y])], axis=0)
                    break

        self.x = points[:,0]
        self.y = points[:,1]
        
        # 預設感測器的目前資料量為0
        self.calc_amount = [0] * self.n_stops
        
    def set_isolated_node(self):
        self.isolated_node = []

        for i in range(self.n_stops):
            is_isolated = None
            for j in range(self.n_stops):
                if i != j:
                    is_isolated = ((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[i]) ** 2 ) ** 0.5 > self.point_range
                if is_isolated == False: 
                    break
            self.isolated_node.append(is_isolated)
        
    # 產生感測器的目前資料量的假資料 max = 100
    def generate_data(self):
        arr1 = self.calc_amount
        arr2 = np.random.randint(self.data_generatation_range, size=self.max_box)

        self.calc_amount = [x + y for x, y in zip(arr1, arr2)]

    def get_unvisited_stops(self):
        # 使用 set 運算來找出未被包含在 route 中的車站
        unvisited_stops = set(list(range(0, self.max_box))) - set(self.stops)
        # 將 set 轉換回 list，方便使用者閱讀
        return list(unvisited_stops)
    
    # 清除無人跡拜訪後的感測器資料
    def clear_data(self):
        for i in self.stops:
            self.calc_amount[i] = 0

    def _generate_q_values(self,box_size = 0.2):
        xy = np.column_stack([self.x,self.y])
        self.q_stops = cdist(xy,xy)


    def render(self,return_img = False):
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")
        
        # Show stops
        ax.scatter(self.x,self.y,c = "black",s = 50)
        
        self.red_stops = []

        # 感測器的資料量大於50，將節點標記為黃色(代表優先節點)
        for i in range(self.n_stops):
            if self.calc_amount[i] > self.calc_threshold:
                ax.scatter(self.x[i], self.y[i], c = "yellow", s = 50)

            if self.calc_amount[i] > self.calc_danger_threshold:
                self.red_stops.append(i)
                ax.scatter(self.x[i], self.y[i], c = "red", s = 50) 
        
        # 將孤立節點標記為灰色
        for i, node in enumerate(self.isolated_node):
            if node == True:
                ax.scatter(self.x[i], self.y[i], c = "#AAAAAA", s = 50)

        # Show START
        if len(self.stops) > 0:
            xy = self._get_xy(initial = True)
            xytext = xy[0] + 0.1, xy[1]-0.05
            ax.annotate("START",xy=xy,xytext=xytext,weight = "bold")

        # Show itinerary
        if len(self.stops) > 1:
            x = np.concatenate((self.x[self.stops], [self.x[self.stops[0]]]))
            y = np.concatenate((self.y[self.stops], [self.y[self.stops[0]]]))
            ax.plot(x, y, c = "blue",linewidth=1,linestyle="--")
            
            # Annotate END
            xy = self._get_xy(initial = False)
            xytext = xy[0]+0.1,xy[1]-0.05
            ax.annotate("END",xy=xy,xytext=xytext,weight = "bold")


        plt.xticks([])
        plt.yticks([])
        
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
        first_stop = 1
        self.stops.append(first_stop)

        return first_stop


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
            self.x[i] = self.x[i] + random.uniform(-self.drift_range, self.drift_range)  # 在-1到1之間的範圍內隨機移動
            self.y[i] = self.y[i] + random.uniform(-self.drift_range, self.drift_range)  # 在-1到1之間的範圍內隨機移動

    def _get_state(self):
        return self.stops[-1]


    def _get_xy(self,initial = False):
        state = calcNodeToOriginDistance(self) if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x,y

    def _get_reward(self,state,new_state):
        distance_reward = self.q_stops[state,new_state]

        has_calc_threshold = self.calc_amount[new_state] > self.calc_threshold
        has_calc_danger_threshold = self.calc_amount[new_state] > self.calc_threshold

        calc_reward = has_calc_threshold * 0.02
        calc_danger_reward = has_calc_danger_threshold * 6
        
        trade_of_factor = 0.001
        


        return (1 - trade_of_factor * distance_reward ** 2) + calc_reward + calc_danger_reward
        # return (1 - trade_of_factor * distance_reward ** 2)
        # return -distance_reward
