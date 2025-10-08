import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib.widgets import Button
import heapq
import random

# ----------------------
# 1) Create graph with 24 nodes
# ----------------------
G = nx.DiGraph()
edges = [
    ("A","B"),("A","C"),("B","D"),("B","E"),("B","F"),
    ("C","G"),("C","H"),("D","I"),("D","J"),("E","K"),
    ("F","L"),("G","M"),("G","N"),("H","O"),("H","P"),
    ("I","Q"),("J","R"),("K","S"),("L","T"),("M","U"),
    ("N","V"),("O","W"),("P","X")
]

# Add edges with random weights
for u, v in edges:
    G.add_edge(u, v, weight=random.randint(1, 10))

# Assign random heuristics
heuristic = {node: random.randint(1, 10) for node in G.nodes()}
heuristic["X"] = 0  # goal heuristic is always 0

# Hierarchical layout
pos = graphviz_layout(G, prog="dot")

# ----------------------
# 2) A* generator
# ----------------------
start_node = "A"
goal_node = "X"

def astar_path_step_by_step(G, start, goal, heuristic):
    # (f, g, node, path)
    pq = [(heuristic[start], 0, start, [start])]
    visited = set()
    while pq:
        f, g, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        yield path  # yield step for animation

        if node == goal:
            return
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                cost = g + G[node][neighbor]["weight"]
                f_score = cost + heuristic[neighbor]
                heapq.heappush(pq, (f_score, cost, neighbor, path + [neighbor]))

# Create generator
path_iter = astar_path_step_by_step(G, start_node, goal_node, heuristic)
visited_nodes = set()

# ----------------------
# 3) Setup figure and button
# ----------------------
fig, ax = plt.subplots(figsize=(12,8))
plt.subplots_adjust(bottom=0.2)

# Draw initial graph
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, ax=ax, arrows=False)

# Edge labels = weights
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

# Node labels = node + heuristic
node_labels = {n: f"{n}\nh={heuristic[n]}" for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_weight="bold", ax=ax)

# Create button
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
button = Button(ax_button, 'Next Step')

# ----------------------
# 4) Button callback
# ----------------------
def update(event):
    global visited_nodes
    try:
        path = next(path_iter)
        visited_nodes.update(path)

        # Node colors
        colors = []
        for node in G.nodes():
            if node == path[-1]:
                colors.append("orange")      # current node
            elif node in visited_nodes:
                colors.append("lightgreen")  # visited
            else:
                colors.append("lightblue")   # unvisited

        ax.clear()

        nx.draw(G, pos, with_labels=False, node_color=colors, node_size=700, ax=ax, arrows=False)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        node_labels = {n: f"{n}\nh={heuristic[n]}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_weight="bold", ax=ax)

        plt.draw()
    except StopIteration:
        print("A* finished")

button.on_clicked(update)
plt.show()
