import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

import csv_utils

n_episodes = 10 # 訓練次數

def calcDistance(x, y):
    distance = 0
    for i in range(len(x) - 1):
        distance += np.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2)
    
    return distance

# 計算移動距離，是否超過最大限制
def calcRouteDistance(env):
    cost = calcDistance(env.x[env.stops], env.y[env.stops])

    to_start_distance = 0
    if cost > 1:
        to_start_distance = calcDistance(env.x[[env.stops[0], env.stops[-1]]], env.y[[env.stops[0], env.stops[-1]]])
        
    cost += to_start_distance

    return cost


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


        # 計算移動距離，是否超過最大限制 (移動距離 + 節點飄移最大的距離)
        # ==============================
        distance = calcRouteDistance(env) + len(env.stops) * env.drift_max_cost
        if distance > env.max_move_distance:
            break
        # ==============================

        

        
    return env,agent,episode_reward


def run_n_episodes(
    env,
    agent,
    name="training.gif",
    n_episodes=n_episodes,
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
            print(i, maxReward)
            maxReward = episode_reward
            img = env.render(return_img = True)
            maxRewardImg = [img]
            max_reward_stop = env.stops


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
    plt.savefig(f"./result/{result_index}_epsilon_min{train_params['epsilon_min']}_loop_index{loop_index}_rewards.png")
    plt.close('all')

    # Save imgs as gif
    # imageio.mimsave(name,imgs,fps = fps)
    imageio.mimsave(f"./result/{result_index}_epsilon_min{train_params['epsilon_min']}_loop_index{loop_index}_qlearning_result.gif",[maxRewardImg[0]],fps = fps)

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

    def optimal_route(route, env, distance):
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

    def get_unvisited_stops(route, env):
        # 使用 set 運算來找出未被包含在 route 中的車站
        unvisited_stops = set(list(range(0, env.max_box))) - set(route)
        # 將 set 轉換回 list，方便使用者閱讀
        return list(unvisited_stops)
    
    # red_stops_distance ======================================
    route,cost = optimal_route(env.red_stops, env, np.Inf)
    red_stops_distance = calcDistance(env.x[route], env.y[route])
    # red_stops_distance ======================================

    # qlearning_distance ======================================
    env.stops = max_reward_stop
    qlearning_distance = calcDistance(env.x[env.stops], env.y[env.stops])
    # qlearning_distance ======================================
    print('\n')
    
    # optimal distance ======================================
    route,cost = optimal_route(env.stops, env, qlearning_distance)
    env.stops = route
    opt_distance = calcDistance(env.x[env.stops], env.y[env.stops])
    # optimal distance ======================================
    print('\n')

    env.unvisited_stops = get_unvisited_stops(route, env)
    
    csv_data = csv_utils.read('./result/train_table.csv')
    csv_data = csv_data + [[red_stops_distance,qlearning_distance,opt_distance]]
    csv_utils.write('./result/train_table.csv', csv_data)

    twoOpt_img = env.render(return_img = True)
    imageio.mimsave(f"./result/{result_index}_epsilon_min{train_params['epsilon_min']}_loop_index{loop_index}_result.gif",[twoOpt_img],fps = fps)

    return env,agent
