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

def calcDistance(x, y):
    distance = 0
    for i in range(len(x) - 1):
        distance += np.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2)
    
    distance += np.sqrt((x[0] - x[-1]) ** 2 + (y[0] - y[-1]) ** 2)
    
    return distance

class QAgent():
    def __init__(self,states_size,actions_size,epsilon = 1.0,epsilon_min = 0.1,epsilon_decay = 0.999,gamma = 1,lr = 0.95):
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
        self.method = method

        # Generate stops
        # self._generate_constraints(**kwargs)
        self._generate_stops()
        self._generate_q_values()
        self.render()

        # Initialize first point
        self.reset()


    def _generate_constraints(self,box_size = 0.2,traffic_intensity = 5):

        if self.method == "traffic_box":

            x_left = np.random.rand() * (self.max_box) * (1-box_size)
            y_bottom = np.random.rand() * (self.max_box) * (1-box_size)

            x_right = x_left + np.random.rand() * box_size * self.max_box
            y_top = y_bottom + np.random.rand() * box_size * self.max_box

            self.box = (x_left,x_right,y_bottom,y_top)
            self.traffic_intensity = traffic_intensity 



    def _generate_stops(self):

        # Generate geographical coordinates
        xy = np.random.rand(self.n_stops,2)*self.max_box
        # print(xy)

        self.x = xy[:,0]
        self.y = xy[:,1]
        
        print(self.x, self.y)

        # self.x = constants.xPoints20
        # self.y = constants.yPoints20

        self.priority_points = np.array(random.sample(list(np.arange(0,self.n_stops)), self.n_stops))
        self.priority_points = self.priority_points[:len(self.priority_points)// 4]


    def _generate_q_values(self,box_size = 0.2):

        # Generate actual Q Values corresponding to time elapsed between two points
        if self.method in ["distance","traffic_box"]:
            xy = np.column_stack([self.x,self.y])
            self.q_stops = cdist(xy,xy)
        else:
            raise Exception("Method not recognized")


    def render(self,return_img = False):
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")

        priority_x = self.x[self.priority_points]
        priority_y = self.y[self.priority_points]

        # Show stops
        ax.scatter(self.x,self.y,c = "red",s = 50)
        # 優先節點畫圖
        # ax.scatter(priority_x,priority_y,c = "yellow",s = 50)


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
            calcDistance(x, y)
            
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
        first_stop = np.random.randint(self.n_stops)
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
        base_reward = self.q_stops[state,new_state]
        return -base_reward
        # extra_reward = 1000
        # additional reaward for priority points
        # print(self.priority_points, new_state, new_state in self.priority_points)
        # if new_state in self.priority_points:
        #     return -base_reward + extra_reward
        # else:
        #     return -base_reward

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
        
        x = np.concatenate((env.x[env.stops], [env.x[env.stops[0]]]))
        y = np.concatenate((env.y[env.stops], [env.y[env.stops[0]]]))
        distance = calcDistance(x,y)


    return env,agent,episode_reward,distance






class DeliveryQAgent(QAgent):

    def __init__(self,*args,**kwargs):
        super().__init__(100, 100)
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



def run_n_episodes(env,agent,name="training.gif",n_episodes=1000,render_each=10,fps=10):

    # Store the rewards
    rewards = []
    # Store the max rewards
    minDistance = np.inf

    minDistanceImg = []

    imgs = []
    # Experience replay
    for i in tqdm(range(n_episodes)):

        # Run the episode
        env,agent,episode_reward,distance = run_episode(env,agent,verbose = 0)
        rewards.append(episode_reward)
        
        if i % render_each == 0:
            img = env.render(return_img = True)
            imgs.append(img)

            if distance < minDistance:
                print('distance', distance, 'episodes', i)
                minDistance = distance
                print(minDistance)
                minDistanceImg.append(img)

    # Show rewards
    plt.figure(figsize = (15,3))
    plt.title("Rewards over training")
    plt.plot(rewards)
    plt.show()

    # Save imgs as gif
    imageio.mimsave(name,imgs,fps = fps)
    imageio.mimsave('result.gif',[minDistanceImg[-1]],fps = fps)


    return env,agent

env,agent = run_n_episodes(DeliveryEnvironment(100, 100), DeliveryQAgent(QAgent))
# Run the episode
env,agent,episode_reward,distance = run_episode(env,agent,verbose = 0)
env.render(return_img = False)