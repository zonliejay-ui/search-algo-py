import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx
import string
import random
from collections import deque
import heapq

#### UI Setup ####
fig, ax = plt.subplots(3, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [8, 1, 1]})
plt.subplots_adjust(hspace=0.4)

ax_plot, ax_button_row1, ax_button_row2 = ax

ax_plot.axis("off")
ax_button_row1.axis("off")
ax_button_row2.axis("off")

# Row 1 buttons
button_ax1 = plt.axes([0.185, 0.13, 0.12, 0.05])
button_ax2 = plt.axes([0.355, 0.13, 0.12, 0.05])
button_ax3 = plt.axes([0.525, 0.13, 0.12, 0.05])
button_ax4 = plt.axes([0.695, 0.13, 0.12, 0.05])

init_btn = Button(button_ax1, "Initial State")
move_button = Button(button_ax2, "Move People")
back_btn = Button(button_ax3, "<")
next_btn = Button(button_ax4, ">")

# Row 2 buttons
button_ax5 = plt.axes([0.185, 0.05, 0.12, 0.05])
button_ax6 = plt.axes([0.355, 0.05, 0.12, 0.05])
button_ax7 = plt.axes([0.525, 0.05, 0.12, 0.05])
button_ax8 = plt.axes([0.695, 0.05, 0.12, 0.05])

astar_btn = Button(button_ax5, "A*")
greedy_btn = Button(button_ax6, "Greedy")
dfs_btn = Button(button_ax7, "")
dls_btn = Button(button_ax8, "")

#### Data Structures ####
nodes = list(['A','B']) #Add nodes
heuristic = {'A': 15,'B': 12} # Add huristics
start = 'A'
goal = 'L'
current_node = 0
visited_nodes = set()
order = []
path = []

pos = {
                   'B': (2, 0),
     'A': (0, -1)
    }

G = nx.Graph()

def render_init_graph(event):
    global order, current_node, start, visited_nodes, path, pos
    ax_plot.clear()
    G.clear()
    start = 'A'
    current_node = 0
    visited_nodes = set()
    order = []
    path = []
    
    G.add_nodes_from(nodes)
    G.add_edge('A','B',cost=3)
    

    pos = {
                   'B': (2, 0), #Add not positions
     'A': (0, -1)
    }

    colors = [ "#77DD77" if n == start else "#FFD1B5" if n == goal else "lightblue" for n in nodes]
    labels = {n: f"{n} ({heuristic[n]})" for n in nodes}
    
    nx.draw(
        G,
        pos,
        labels=labels,
        node_color=colors,
        node_size=600,
        font_weight="bold",
        ax=ax_plot
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=nx.get_edge_attributes(G, "cost"),
        ax=ax_plot,
        font_color="red",
        font_size=10
    )
    fig.canvas.draw_idle()

def move_people(event):
    global order, current_node, start, visited_nodes, path, pos

    if len(order) > 0 and current_node < len(order):
        start = order[current_node-1]
    
    ax_plot.clear()
    G.clear()
    current_node = 0
    visited_nodes = set()
    order = []
    path = []
    
    G.add_nodes_from(nodes)
    G.add_edge('A','B',cost=3)


    pos = {
                   'B': (1, 0),
     'A': (0, -1)
    }

    colors = [ "#77DD77" if n == start else "#FFD1B5" if n == goal else "lightblue" for n in nodes]
    labels = {n: f"{n} ({heuristic[n]})" for n in nodes}
    
    nx.draw(
        G,
        pos,
        labels=labels,
        node_color=colors,
        node_size=600,
        font_weight="bold",
        ax=ax_plot
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=nx.get_edge_attributes(G, "cost"),
        ax=ax_plot,
        font_color="red",
        font_size=10
    )
    fig.canvas.draw_idle()

# A* Search
def a_star(event):
    global order, path, current_node, visited_nodes

    # Priority queue: (f_score, cumulative_cost, node)
    pq = [(heuristic[start], 0, start)]
    parent = {start: None}
    cost_so_far = {start: 0}
    order = []

    found = False
    while pq:
        f_score, current_cost, node = heapq.heappop(pq)

        if node in order:  # Already expanded
            continue

        order.append(node)  # Track visitation

        if node == goal:
            found = True
            break

        for neighbor in G.neighbors(node):
            edge_cost = G[node][neighbor]['cost']
            new_cost = current_cost + edge_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(pq, (new_cost + heuristic[neighbor], new_cost, neighbor))

    # Reconstruct path
    if found:
        node = goal
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

    # Reset for GUI traversal
    current_node = 0
    visited_nodes = set()
    next(None)  # Start GUI animation

# Greedy Best-First Search
def greedy(event):
    global order, path, current_node, visited_nodes

    # Priority queue: (heuristic, node)
    pq = [(heuristic[start], start)]
    parent = {start: None}
    order = []

    found = False
    visited = set()

    while pq:
        h_value, node = heapq.heappop(pq)

        if node in visited:  # Already expanded
            continue

        visited.add(node)
        order.append(node)  # Track visitation

        if node == goal:
            found = True
            break

        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                parent[neighbor] = node
                heapq.heappush(pq, (heuristic[neighbor], neighbor))

    # Reconstruct path
    if found:
        node = goal
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

    # Reset for GUI traversal
    current_node = 0
    visited_nodes = set()
    next(None)  # Start GUI animation

# Next button event handler
def next(event):
    global current_node, visited_nodes

    if current_node >= len(order):
        print("Reached end of order")
        return

    node = order[current_node]
    visited_nodes.add(node)

    # Prepare colors
    colors = []
    for n in nodes:
        if n == node:
            colors.append("orange")  # Current node
        elif n in visited_nodes:
            colors.append("lightgreen")  # Visited nodes
        else:
            colors.append("lightblue")   # Unvisited

    # If goal reached, highlight the final path in red
    if node == goal:
        for n in path:
            colors[nodes.index(n)] = "red"

    re_draw_graph(colors)
    current_node += 1

# Back button event handler
def back(event):
    global current_node, visited_nodes

    if current_node <= 0:
        print("Already at the beginning")
        return

    # Decrement current_node first
    current_node -= 1

    # Remove current node from visited (since we're going back)
    node = order[current_node]
    if node in visited_nodes:
        visited_nodes.remove(node)

    # Prepare colors for all nodes
    full_colors = []
    for n in nodes:
        if n == order[current_node]:
            full_colors.append("orange")        # Current node
        elif n in visited_nodes:
            full_colors.append("lightgreen")    # Visited nodes
        else:
            full_colors.append("lightblue")     # Unvisited nodes

    # Clear and redraw the graph
    re_draw_graph(full_colors)

def re_draw_graph(colors):
    labels = {n: f"{n} ({heuristic[n]})" for n in nodes}
    # Clear and redraw the graph
    ax_plot.clear()
    nx.draw(
        G,
        pos,
        labels=labels,
        node_color=colors,
        node_size=600,
        font_weight="bold",
        ax=ax_plot
    )

    # Draw edge costs
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=nx.get_edge_attributes(G, "cost"),
        ax=ax_plot,
        font_color="red",
        font_size=10
    )

    fig.canvas.draw_idle()


def draw_search_graph(event):
    # Labels
    labels = list(string.ascii_uppercase[:24])

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(labels)

    # Add some edges
    for i in range(0, 100):
        ni = random.randint(0, 23)
        nii = abs(23 - ni)
        G.add_edge(labels[ni], labels[nii])
        G.add_edge(labels[ni], labels[abs(nii-3)])
        G.add_edge(labels[ni], labels[abs(nii-10)])

    # Draw
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=600,
        font_weight="bold",
        ax=ax_plot
    )
    plt.show()

init_btn.on_clicked(render_init_graph)
move_button.on_clicked(move_people)
next_btn.on_clicked(next)
back_btn.on_clicked(back)

astar_btn.on_clicked(a_star)
greedy_btn.on_clicked(greedy)
plt.show()