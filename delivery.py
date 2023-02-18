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
import constants

plt.style.use("seaborn-v0_8-dark")

sys.path.append("../")

minDistance = -np.inf

# 設定環境參數
point_num = 50 # 節點數輛
maxDistance = 200 # 無人機最大移動距離


# 迴圈每次執行的Q-learning參數需不同
# 先簡單分為前500次與後500次

def calcDistance(x, y):
    distance = 0
    for i in range(len(x) - 1):
        distance += np.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2)
    
    return distance

class QAgent():
    def __init__(self,states_size,actions_size,epsilon = 1.0,epsilon_min = 0.05,epsilon_decay = 0.9998,gamma = 0.65,lr = 0.65):
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

        # Generate geographical coordinates
        xy = np.random.rand(self.n_stops,2)*self.max_box

        self.x = xy[:,0]
        self.y = xy[:,1]
        
        # self.x = constants.xPoints20
        # self.y = constants.yPoints20
        
        # 預設感測器的目前資料量為0
        self.calc_amount = [0] * self.n_stops
        
        # 產生感測器的目前資料量的假資料
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



        if hasattr(self,"box"):
            left,bottom = self.box[0],self.box[2]
            width = self.box[1] - self.box[0]
            height = self.box[3] - self.box[2]
            rect = Rectangle((left,bottom), width, height)
            collection = PatchCollection([rect],facecolor = "red",alpha = 0.2)
            ax.add_collection(collection)


        plt.xticks([])
        plt.yticks([])
        
        if return_img:
            # From https://ndres.me/post/matplotlib-animated-gifs-easily/
            fig.canvas.draw()
            # fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
        else:
            plt.show()



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
        calc_danger_reward = has_calc_danger_threshold * 5
        
        trade_of_factor = 0.001
        
        # print(1 - trade_of_factor * distance_reward ** 2)
        
        return (1 - trade_of_factor * distance_reward ** 2) + calc_reward + calc_danger_reward
        # return (1 - trade_of_factor * distance_reward ** 2)
        # extra_reward = 1000
        # additional reaward for priority points
        # print(self.priority_points, new_state, new_state in self.priority_points)
        # if new_state in self.priority_points:
        #     return -distance_reward + extra_reward
        # else:
        #     return -distance_reward

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

        distance = calcDistance(env.x[env.stops], env.y[env.stops])

        to_start_distance = 0
        if i >= 1:
            to_start_distance = calcDistance(env.x[[env.stops[0], env.stops[-1]]], env.y[[env.stops[0], env.stops[-1]]])
            
        distance += to_start_distance

        #  紀錄獎勵最高的圖片
        # if episode_reward > maxReward:
        if distance > maxDistance:
            break

        
    return env,agent,episode_reward






class DeliveryQAgent(QAgent):

    def __init__(self,*args,**kwargs):
        super().__init__(point_num, point_num)
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




def run_n_episodes(env,agent,name="training.gif",n_episodes=10000,render_each=10,fps=10):

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
            maxRewardImg.append(img)
            max_reward_stop = env.stops
            print('maxReward', maxReward, 'index', i)
                

        # 當執行迴圈到一半時，更改參數
        if i == (n_episodes // 2):
            # agent.gamma = 0.45
            # agent.lr = 0.65
            # agent.epsilon = 0.1
            # agent.epsilon_min = 0.1
            imageio.mimsave('pre_result.gif',[maxRewardImg[-1]],fps = fps)

    # Show rewards
    plt.figure(figsize = (15,3))
    plt.title("Rewards over training")
    plt.plot(rewards)
    plt.show()

    # Save imgs as gif
    # imageio.mimsave(name,imgs,fps = fps)
    imageio.mimsave('qlearning_result.gif',[maxRewardImg[-1]],fps = fps)

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
    
    route,cost = optimalRoute(env.red_stops, env, np.Inf)
    red_stops_distance = calcDistance(env.x[route], env.y[route])
    
    print('red_stops', route)
    print('red_stops_distance', red_stops_distance)

    
    env.stops = max_reward_stop
    qlearning_distance = calcDistance(env.x[env.stops], env.y[env.stops])
    print('stops', max_reward_stop)
    print('qlearning distance', qlearning_distance)
    
    route,cost = optimalRoute(env.stops, env, qlearning_distance)
    env.stops = route
    print('2opt stops', route)
    print('result distance', calcDistance(env.x[env.stops], env.y[env.stops]))
    
    
    twoOpt_img = env.render(return_img = True)
    imageio.mimsave('result.gif',[twoOpt_img],fps = fps)

    return env,agent

env,agent = run_n_episodes(DeliveryEnvironment(point_num, 50), DeliveryQAgent(QAgent))
# Run the episode
env,agent,episode_reward = run_episode(env,agent,verbose = 0)
env.render(return_img = False)
