import math

# 取得感測器的周圍檢點資訊
# 將所有感測器跑回圈
# 判斷哪些感測器有計算後的資料
# 將資料丟到附近距離最近的感測器
# 並且將資料從原本的感測器中刪除
# 並且將資料加入到新的感測器中
# 新的感測器掃描周圍的感測器
# 並且將資料丟到附近距離最近的感測器
# 直到無法再丟感測器了
# index + 1

point_arr = []

class GPSR_Node:
    def __init__(self, x, y, i, env):
        self.x = x
        self.y = y
        self.env = env
        self.index = i
        self.around_nodes = []
        self.first_point = env.first_point
        self.nearest_sink_node = None
        self.parent_num = 0
        
        # method
        self.get_around_nodes()
        self.get_nearest_sink_node()
        
    def euclidean_distance(self, node):
        sink_x = self.env.x[self.first_point]
        sink_y = self.env.y[self.first_point]
        
        return math.sqrt((sink_x - node['x']) ** 2 + (sink_y - node['y']) ** 2)
    
    def get_around_nodes(self):
        self.around_nodes = []
        
        current_distance = self.euclidean_distance({ 'x': self.x, 'y': self.y })
        
        for idx, x in enumerate(self.env.x):
            x = self.env.x[idx]
            y = self.env.y[idx]
            
            sink_distance = self.euclidean_distance({ 'x': x, 'y': y})
            nearby_distance = math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
            
            if idx != self.index and nearby_distance <= self.env.communication_range and sink_distance < current_distance:
                self.around_nodes.append(idx)
        
        return self.around_nodes 

    def get_nearest_sink_node(self):
        self.nearest_sink_node = None
        nearest_distance = None
        
        for node in self.around_nodes:
            x = self.env.x[self.first_point]
            y = self.env.y[self.first_point]
            
            distance = self.euclidean_distance({ 'x': x, 'y': y})
            if nearest_distance == None or distance < nearest_distance:
                self.nearest_sink_node = node
                nearest_distance = distance
        
        
def generate_gpsr_node(env):
    arr = []
    for idx, x in enumerate(env.x):
        x = env.x[idx]
        y = env.y[idx]
        arr.append(GPSR_Node(x, y, idx, env))
        
    return arr

def set_tree_parent_num(env):
    arr = generate_gpsr_node(env)
    
    for idx, current_node in enumerate(arr):
        
        current_index = current_node
        
        while (True):
            nearest_sink_node = current_index.nearest_sink_node
            if nearest_sink_node != None:
                arr[nearest_sink_node].parent_num = arr[nearest_sink_node].parent_num + 1
                
                if env.first_point == nearest_sink_node:
                    print('84', current_index.index, current_index.around_nodes)
                    break
                else:
                    current_index = arr[nearest_sink_node]
            else:
                break
    
    return [node.parent_num  for idx, node in enumerate(arr)]

def run_gpsr_node(env):
    arr = generate_gpsr_node(env)
    
    connect_num = 0
    for idx, current_node in enumerate(arr):
        
        while (True):
            nearest_sink_node = current_node.nearest_sink_node
            if nearest_sink_node != None:
                current_index = current_node.index
                
                if env.first_point == nearest_sink_node:
                    env.sum_mutihop_data = env.sum_mutihop_data + env.data_amount_list[current_index]['calc']
                    env.data_amount_list[current_index]['calc'] = 0
                    connect_num = connect_num + 1
                    break
                else:
                    env.data_amount_list[nearest_sink_node]['calc'] = env.data_amount_list[nearest_sink_node]['calc'] + env.data_amount_list[current_index]['calc']
                env.data_amount_list[current_index]['calc'] = 0

                current_node = arr[nearest_sink_node]
            else:
                break
    
    print('connect_num', connect_num)
    return arr