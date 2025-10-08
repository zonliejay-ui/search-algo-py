# search-algo-py

A simple Python project demonstrating basic search algorithms: **A***, **UCS**, **DFS**, **BFS**, and **Greedy Best-First Search**.

## Features

* Clear and minimal implementations for classic search algorithms.
* Example problems for testing.
* Easy to modify and extend.

## Installation

```bash
git clone https://github.com/<your-username>/search-algo-py.git
cd search-algo-py
```

## Usage

Run an example from the command line:

```bash
python run_search.py --algorithm astar --problem examples/grid_maze.json
```

Or use in Python:

```python
from search_algos import astar
path = astar.search(graph, start, goal)
print(path)
```

## Project Structure

```
search-algo-py/
├── run_search.py
├── search_algos/
│   ├── astar.py
│   ├── bfs.py
│   ├── dfs.py
│   ├── greedy.py
│   └── ucs.py
└── examples/
```

## License

MIT License.
