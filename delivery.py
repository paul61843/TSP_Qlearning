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

plt.style.use("seaborn-v0_8-dark")

sys.path.append("../")

# 待完成事項
# 1. 須建立 tree (或是 k-means) 決定，感測器的回傳sink的資料傳輸路徑
# 2. 跑到每一個點後，需計算無人機剩餘的電量，決定是否添加新的拜訪點
# 3. 跑完一輪後，節點飄移功能需完成，並且飄移後的節點是否成為斷路的判斷
# 4. 飄移後的節點，因離開原始位置，無人機需增加搜尋功能，找尋漂離的節點
# 

# 設定環境參數
mutliprocessing_num = 1 # 產生結果數量
point_num = 50 # 節點數
point_radius = 10 # 節點通訊範圍 (單位m)
max_distance = 200 # 無人機最大移動距離 (單位m)
drift_distance = 5

n_episodes = 2 # 訓練次數

def calcDistance(x, y):
    distance = 0
    for i in range(len(x) - 1):
        distance += np.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2)
    
    return distance

class QAgent():
    def __init__(self,states_size,actions_size,epsilon,epsilon_min,epsilon_decay,gamma,lr):
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q = self.build_model(states_size,actions_size)


    def build_model(self,states_size,actions_size):
        Q = np.zeros([states_size,actions_size])
        return Q


    def train(self,s,a,r,s_next):
        self.Q[s,a] = self.Q[s,a] + self.lr * (r + self.gamma*np.max(self.Q[s_next,a]) - self.Q[s,a])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def act(self,s):

        q = self.Q[s,:]

        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            a = np.random.randint(self.actions_size)

        return a



class DeliveryEnvironment(object):
    def __init__(self,n_stops = 20,max_box = 10,method = "distance",**kwargs):

        print(f"Initialized Delivery Environment with {n_stops} random stops")
        print(f"Target metric for optimization is {method}")

        # Initialization
        self.n_stops = n_stops
        self.action_space = self.n_stops
        self.observation_space = self.n_stops
        self.max_box = max_box
        self.stops = []
        self.red_stops = []
        self.method = method
        self.calc_threshold = 50
        self.calc_danger_threshold = 75

        # Generate stops
        # self._generate_constraints(**kwargs)
        self._generate_stops()
        self._generate_q_values()
        self.render()

        # Initialize first point
        self.reset()

    def _generate_stops(self):
        points = np.random.rand(1,2) * self.max_box

        # 隨機生成感測器數量，並確保每個點的通訊範圍內至少有一個點
        while (len(points) < self.n_stops):
            x,y = (np.random.rand(1,2) * self.max_box)[0]
            for p in points:
                isTrue = any(((x - p[0]) ** 2 + (y - p[1]) ** 2 ) ** 0.5 <= point_radius for p in points)
                if isTrue:
                    points = np.append(points, [np.array([x,y])], axis=0)
                    break

        # Generate geographical coordinates
        # xy = np.random.rand(self.n_stops,2) * self.max_box

        self.x = points[:,0]
        self.y = points[:,1]
        
        # self.x = constants.xPoints20
        # self.y = constants.yPoints20
        
        # 預設感測器的目前資料量為0
        self.calc_amount = [0] * self.n_stops
        
        # 產生感測器的目前資料量的假資料 max = 100
        self.calc_amount = np.random.randint(100, size=self.max_box)
        
        
        # self.priority_points = np.array(random.sample(list(np.arange(0,self.n_stops)), self.n_stops))
        # self.priority_points = self.priority_points[:len(self.priority_points)// 4]


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
            
        # Show START
        if len(self.stops) > 0:
            xy = self._get_xy(initial = True)
            xytext = xy[0]+0.1,xy[1]-0.05
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
        # Random first stop
        # first_stop = np.random.randint(self.n_stops)
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
            self.x[i] = self.x[i] + random.uniform(-drift_distance, drift_distance)  # 在-1到1之間的範圍內隨機移動
            self.y[i] = self.y[i] + random.uniform(-drift_distance, drift_distance)  # 在-1到1之間的範圍內隨機移動

    def _get_state(self):
        return self.stops[-1]


    def _get_xy(self,initial = False):
        state = self.stops[0] if initial else self._get_state()
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
        # return -distance_reward ** 2


    @staticmethod
    def _calculate_point(x1,x2,y1,y2,x = None,y = None):

        if y1 == y2:
            return y1
        elif x1 == x2:
            return x1
        else:
            a = (y2-y1)/(x2-x1)
            b = y2 - a * x2

            if x is None:
                x = (y-b)/a
                return x
            elif y is None:
                y = a*x+b
                return y
            else:
                raise Exception("Provide x or y")

    def _calculate_box_intersection(self,x1,x2,y1,y2,box):
        return 'intersections'

def run_episode(env,agent,verbose = 1):

    s = env.reset()
    agent.reset_memory()

    max_step = env.n_stops
    
    episode_reward = 0
    
    i = 0

    while i < max_step:
        # Remember the states
        agent.remember_state(s)

        # Choose an action
        a = agent.act(s)

        # Take the action, and get the reward from environment
        s_next,r,done = env.step(a)

        if verbose: print(s_next,r,done)
        
        # Update our knowledge in the Q-table
        agent.train(s,a,r,s_next)
        # Update the caches
        episode_reward += r
        s = s_next
        
        # If the episode is terminated
        i += 1
        if done:
            break


        # 計算移動距離，是否超過最大限制
        # ==============================
        distance = calcDistance(env.x[env.stops], env.y[env.stops])

        to_start_distance = 0
        if i >= 1:
            to_start_distance = calcDistance(env.x[[env.stops[0], env.stops[-1]]], env.y[[env.stops[0], env.stops[-1]]])
            
        distance += to_start_distance

        if distance > max_distance:
            break
        # ==============================

        

        
    return env,agent,episode_reward



class DeliveryQAgent(QAgent):

    def __init__(self,*args,**kwargs):
        super().__init__(**kwargs)
        self.reset_memory()

    def act(self,s):

        # Get Q Vector
        q = np.copy(self.Q[s,:])

        # Avoid already visited states
        q[self.states_memory] = -np.inf

        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            a = np.random.choice([x for x in range(self.actions_size) if x not in self.states_memory])

        return a


    def remember_state(self,s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []




def run_n_episodes(
    env,
    agent,
    name="training.gif",
    n_episodes=n_episodes,
    render_each=10,
    fps=10,
    result_index=1,
    train_params={},
):

    # Store the rewards
    rewards = []
    # Store the max rewards
    maxReward = -np.inf

    maxRewardImg = []
    
    max_reward_stop = []

    imgs = []

    # Experience replay
    for i in tqdm(range(n_episodes)):

        # Run the episode
        env,agent,episode_reward = run_episode(env,agent,verbose = 0)
        rewards.append(episode_reward)

        if i % render_each == 0:
            img = env.render(return_img = True)
            imgs.append(img)

        #  紀錄獎勵最高的圖片
        if episode_reward > maxReward:
            maxReward = episode_reward
            img = env.render(return_img = True)
            maxRewardImg = [img]
            max_reward_stop = env.stops
            print('maxReward', maxReward, 'index', i)


        # 當執行迴圈到一半時，更改參數
        # if i == (n_episodes // 2):
            # agent.gamma = 0.45
            # agent.lr = 0.65
            # agent.epsilon = 0.1
            # agent.epsilon_min = 0.1
            # imageio.mimsave('pre_result.gif',[maxRewardImg[-1]],fps = fps)

    # Show rewards
    plt.figure(figsize = (15,3))
    plt.title("Rewards over training")
    plt.plot(rewards)
    plt.savefig(f"./result/epsilon_min_{train_params['epsilon_min']}_{result_index}_rewards.png")
    plt.close('all')

    # Save imgs as gif
    # imageio.mimsave(name,imgs,fps = fps)
    imageio.mimsave(f"./result/epsilon_min_{train_params['epsilon_min']}_{result_index}_qlearning_result.gif",[maxRewardImg[0]],fps = fps)

    # 2-opt 程式碼
    def swap(route,i,k):
        new_route = []
        for j in range(0,i):
            new_route.append(route[j])
        for j in range(k,i-1,-1):
            new_route.append(route[j])
        for j in range(k+1,len(route)):
            new_route.append(route[j])
        return new_route

    def optimalRoute(route, env, distance):
        cost = distance
        for i in range(1000):
            for j in range(len(route)):
                for k in range(len(route)):
                    if j < k:
                        new_route = swap(route,j,k)
                        new_cost = calcDistance(env.x[new_route], env.y[new_route])
                        if new_cost < cost:
                            route = new_route
                            cost = new_cost
        return route,cost
    # 2-opt 程式碼 end
    
    # red_stops_distance ======================================
    route,cost = optimalRoute(env.red_stops, env, np.Inf)
    red_stops_distance = calcDistance(env.x[route], env.y[route])
    # red_stops_distance ======================================

    
    # qlearning_distance ======================================
    env.stops = max_reward_stop
    qlearning_distance = calcDistance(env.x[env.stops], env.y[env.stops])
    # qlearning_distance ======================================
    print('\n')
    
    # result distance ======================================
    route,cost = optimalRoute(env.stops, env, qlearning_distance)
    env.stops = route
    opt_distance = calcDistance(env.x[env.stops], env.y[env.stops])
    # result distance ======================================
    print('\n')
    
    csv_data = csv_utils.read('./result/train_table.csv')
    csv_data = csv_data + [[red_stops_distance,qlearning_distance,opt_distance]]
    csv_utils.write('./result/train_table.csv', csv_data)

    twoOpt_img = env.render(return_img = True)
    imageio.mimsave(f"./result/epsilon_min_{train_params['epsilon_min']}_{result_index}_result.gif",[twoOpt_img],fps = fps)

    return env,agent

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
        env,agent = run_n_episodes(
            DeliveryEnvironment(point_num, 50), 
            DeliveryQAgent(
                states_size=point_num,
                actions_size=point_num,
                epsilon = 1.0,
                epsilon_min = params["epsilon_min"],
                epsilon_decay = 0.9998,
                gamma = 0.65,
                lr = 0.65
            ),
            result_index=index,
            train_params=params,
        )
        # Run the episode
        env,agent,episode_reward = run_episode(env,agent,verbose = 0)
        env.render(return_img = True)
        env.drift_node()
        print(f'run {index} end ========================================')


# mutiprocessing start ================================
if __name__ == '__main__':
    process_list = []
    csv_utils.write('./result/train_table.csv', 
        [['red_distance','q_distance','opt_distance']]
    )

    for i in range(mutliprocessing_num):
        process_list.append(mp.Process(target=runMain, args=(i,)))
        process_list[i].start()

    for i in range(mutliprocessing_num):
        process_list[i].join()

# mutiprocessing end ================================