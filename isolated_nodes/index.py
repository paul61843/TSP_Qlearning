import random
import math
from collections import defaultdict, deque

# 计算两个节点之间的欧氏距离
def calculate_distance(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 构建节点之间的连接关系图
def build_graph(nodes, sensing_range):
    graph = defaultdict(list)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            distance = calculate_distance(node1, node2)
            if distance <= sensing_range:
                graph[node1].append(node2)
                graph[node2].append(node1)
    return graph

# 使用广度优先搜索查找从起始节点到目标节点的路径
def bfs(graph, start, target, sensing_range):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        node, path = queue.popleft()
        distance = calculate_distance(node, target)

        if distance <= sensing_range:
            return path


        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# 查找孤立节点
def find_isolated_nodes(nodes, sensing_range, sink):
    graph = build_graph(nodes, sensing_range)
    isolated_nodes = []
    for idx, node in enumerate(nodes):
    # for node in nodes:
        if bfs(graph, node, sink, sensing_range) is None:
            isolated_nodes.append(idx)

    return isolated_nodes

# 查找孤立节点
# isolated_nodes = find_isolated_nodes(node_coordinates, sensing_range, sink)

