import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
import math
import computerVision.obstacles as obs
import cv2
import camera

# to know if the segment is on an obstacle 
def intersects_obstacle(p1, p2, obstacles):
    line = LineString([p1, p2])
    for obstacle in obstacles:
        if (line.within(obstacle) or line.crosses(obstacle)):
            return True
    
    return False
    
def is_in_bounds(node, bounds):
    x_upper, x_lower, y_upper, y_lower = bounds
    if node[0] < x_lower or node[0] > x_upper or node[1] < y_lower or node[1] > y_upper:
        return False
    else:
        return True

# visibility graph of all the possible paths from start to goal
def build_visibility_graph(start, goal, extended_obstacles, bounds):
    G = nx.Graph()
    nodes = [start,goal]
    for extended_obstacle in extended_obstacles:
        if isinstance(extended_obstacle, Polygon):
            obstacle_corners = list(extended_obstacle.exterior.coords[:-1])
            good_corners = []
            for corner in obstacle_corners:
                print(corner)
                if is_in_bounds(corner, bounds):
                    good_corners.append(corner)
            nodes.extend(good_corners) #the obstacle corners that are in the frame are added as nodes of the visibility graph
        elif isinstance(extended_obstacle, MultiPolygon):
            for poly in extended_obstacle.geoms:
                obstacle_corners = list(poly.exterior.coords[:-1])
                for corner in obstacle_corners:
                    print(corner)
                    if is_in_bounds(corner, bounds):
                        good_corners.append(corner)
                nodes.extend(good_corners) #the obstacle corners that are in the frame are added as nodes of the visibility graph
    
    for i, node in enumerate(nodes):
        G.add_node(i, pos=node)
    
    for i, p1 in enumerate(nodes):
        for j, p2 in enumerate(nodes):
            if i != j and not intersects_obstacle(p1, p2,extended_obstacles):
                distance = np.linalg.norm(np.array(p1) - np.array(p2))
                G.add_edge(i, j, weight=distance) #the segments that are not crossing obstacles are added to the visibility graph
    
    return G, nodes

# to find the optimal path among all the paths of the visibility graph
def dijkstra_search(G, start_idx, goal_idx):
    
    distances = {node: math.inf for node in G.nodes}  
    distances[start_idx] = 0  
    
    
    predecessors = {node: None for node in G.nodes}  
    visited = set()  

    while len(visited) < len(G.nodes):
        
        unvisited_nodes = [node for node in distances if node not in visited]
        current_node = min(unvisited_nodes, key=lambda node: distances[node])
        
        visited.add(current_node)

        for neighbor in G.neighbors(current_node):
            if neighbor not in visited:
                edge_weight = G[current_node][neighbor].get('weight', 1) 
                new_distance = distances[current_node] + edge_weight  
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node   
                
        if current_node == goal_idx:
            break            

    path = []
    while current_node in predecessors:
        path.append(current_node)
        current_node = predecessors[current_node]
    
    return path[::-1]   

#plotting
def plot_map_and_path(G, nodes, obstacles, path): 
    fig, ax = plt.subplots()  
    
    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.7)
    
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_size=5, node_color='blue', ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, ax=ax)
        
    ax.plot(*nodes[0], "go", label="Start")
    ax.plot(*nodes[1], "ro", label="Goal")
    plt.legend()
    plt.show()

def find_optimal_path(start,goal,obstacles, size_robot, bounds):

    extended_obstacles = [obstacle.buffer(size_robot, join_style = 'mitre', mitre_limit = 1) for obstacle in obstacles]
    
       
    G, nodes = build_visibility_graph(start, goal, extended_obstacles, bounds)

    start_idx = 0  
    goal_idx = 1  
    try:
        path = dijkstra_search(G, start_idx, goal_idx)
    except nx.NetworkXNoPath:
        path = None

    return G,nodes,path

def image_to_graph_polygones(polygons, image_shape):
    
    height = image_shape[1]
    transformed_polygons = []
    
    for polygon in polygons:
        transformed_coords = [
            (x, height - y) for x, y in polygon.exterior.coords
        ]
        transformed_polygon = Polygon(transformed_coords)
        transformed_polygons.append(transformed_polygon)
    
    return transformed_polygons

#global function to find the optimal path
def get_optimal_path(size_robot, start, goal, obstacle_polygons, bounds):   


    G, nodes, path = find_optimal_path(start, goal, obstacle_polygons, size_robot, bounds)   

    
    print("None appended path:", path)
    
    path = [nodes[idx] for idx in path]
    return [(int(node[0]), int(node[1])) for node in path]

