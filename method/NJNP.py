import numpy as np

def run_NJNP(env, parent_num):
    
    priority_nodes = []
    
    sink_x = env.x[env.first_point]
    sink_y = env.y[env.first_point]
    
    for idx, x in enumerate(env.x):
        if idx == env.first_point:
            continue
        
        x = env.x[idx]
        y = env.y[idx]
        parent = parent_num[idx]
        
        distance = ((x - sink_x) ** 2 + (y - sink_y)**2) ** 0.5
        
        priority_nodes.append(parent / distance)
    
    priority_nodes = np.array(priority_nodes)
    
    priority_nodes_index = sorted(range(len(priority_nodes)), key=lambda k: priority_nodes[k], reverse=True) 
    
    return list(priority_nodes_index)
        