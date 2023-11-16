import numpy as np

def getNJNPNextPoint(env, child_nodes):
    max_value = -1
    max_node = None
    
    current_node = env.stops[-1]

    remain_unconnect_nodes = [elem for elem in env.unconnect_nodes if elem not in list(env.stops)]

    for i in remain_unconnect_nodes:

        if current_node != i:    
            child_num = child_nodes[i]

            distance = ((env.x[i] - env.x[current_node]) ** 2 + (env.y[i] - env.y[current_node])**2) ** 0.5

            data_amount = child_num / distance

            if data_amount >= max_value:
                max_value = data_amount
                max_node = i

    return max_node
    

def run_NJNP(env, child_num):
    
    priority_nodes = []
    
    sink_x = env.x[env.first_point]
    sink_y = env.y[env.first_point]
    
    for idx, x in enumerate(env.x):
        if idx == env.first_point:
            continue
        
        x = env.x[idx]
        y = env.y[idx]
        child = child_num[idx]
        
        distance = ((x - sink_x) ** 2 + (y - sink_y)**2) ** 0.5
        
        priority_nodes.append(child / distance)
    
    priority_nodes = np.array(priority_nodes)
    
    priority_nodes_index = sorted(range(len(priority_nodes)), key=lambda k: priority_nodes[k], reverse=True) 
    
    return list(priority_nodes_index)
        