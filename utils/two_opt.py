from utils.calc import *
import numpy as np
import random

# # 2-opt 程式碼
def optimal_route(route, env, distance):
    cost = distance
    if len(route) == 0:
        return route,cost

    for i in range(1000):
        new_route = list(route)
        if i % 2 == 0: 
            new_route = swapMore(new_route)
        else:
            new_route = swapOne(new_route)

        new_cost = calcDistance(env.x[new_route], env.y[new_route])

        if new_cost < cost:
            route = new_route
            cost = new_cost
        
        
    return route,cost
# # 2-opt 程式碼 end

def swapMore(route):
    index1 = random.choice(list(range(len(route))))
    index2 = random.choice(list(range(index1, len(route))))

    for i in range((index2 - index1) // 2):
        route = swap(route, index1+i, index2-i)
    return route

def swapOne(route):
    index1 = random.choice(range(len(route)))
    index2 = random.choice(range(len(route)))

    temp = route[index1]
    route[index1] = route[index2]
    route[index2] = temp

    return route

def swap(route, index1, index2):
	temp = route[index1]
	route[index1] = route[index2]
	route[index2] = temp
	return route

# 2-opt 程式碼
# def swap(route,i,k):
#     new_route = []
#     for j in range(0,i):
#         new_route.append(route[j])
#     for j in range(k,i-1,-1):
#         new_route.append(route[j])
#     for j in range(k+1,len(route)):
#         new_route.append(route[j])
#     return new_route

# def optimal_route(route, env, distance):
#     cost = distance
#     for i in range(5000):
#         for j in range(len(route)):
#             for k in range(len(route)):
#                 if j < k:
#                     new_route = swap(route,j,k)
#                     new_cost = calcDistance(env.x[new_route], env.y[new_route])
#                     if new_cost < cost:
#                         route = new_route
#                         cost = new_cost
#     return route,cost
    # 2-opt 程式碼 end
    