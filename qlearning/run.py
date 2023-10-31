import imageio
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

import csv_utils

from utils.calc import *
from utils.two_opt import *


def run_episode(env,agent,verbose = 1):

    env.reset()
    s = env.first_point
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


        env.drift_cost_list = len(env.stops) * [env.drift_max_cost]
        # 計算移動距離，是否超過最大限制 (移動距離 + 節點飄移最大的距離)
        # Q learning 跑出的結果 會超過最大移動距離，減 20 避免超過
        # ==============================
        distance = calcPowerCost(env)
        if distance > env.max_move_distance - 20:
            break
        # ==============================
        
        
    return env,agent,episode_reward


def run_n_episodes(
    env,
    agent,
    name="training.gif",
    n_episodes=1000,
    render_each=10,
    fps=10,
    result_index=0,
    loop_index=0,
    train_params={},
):

    # Store the rewards
    rewards = []
    # Store the max rewards
    maxReward = -np.inf

    max_reward_stop = []

    # Experience replay
    for i in tqdm(range(n_episodes)):
        
        # Run the episode
        env,agent,episode_reward = run_episode(env,agent,verbose = 0)
        rewards.append(episode_reward)

        #  紀錄獎勵最高的圖片
        if episode_reward > maxReward:
            maxReward = episode_reward
            # img = env.render(return_img = True)
            # maxRewardImg = [img]
            max_reward_stop = env.stops
            
            cost = calcPowerCost(env)
            distance = calcDistance(env.x[env.stops], env.y[env.stops])
            print('99', cost, distance, sum(env.drift_cost_list))
    
    # Show rewards
    plt.figure(figsize = (15,3))
    plt.title("Rewards over training")
    plt.plot(rewards)
    plt.savefig(f"./result/Q_learning/{result_index}_epsilon_min{train_params['epsilon_min']}_gamma{train_params['gamma']}_lr{train_params['lr']}_loop_index{loop_index}_rewards.png")
    plt.close('all')

    # qlearning_distance ======================================
    env.stops = max_reward_stop
    
    env.unvisited_stops = env.get_unvisited_stops()
    env.remain_power = calcPowerCost(env)
    env.drift_cost_list = len(env.stops) * [env.drift_max_cost]
    
    return env,agent
