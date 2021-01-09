from src.agent import Agent
from src.astar import AStar
import time
from src.common import TIME
from src.dataviz import DataViz
from src.world import World


if __name__ == "__main__":
    # all coordinates are given in format [y,x]
    # import doctest

    # doctest.testmod()

    w = World(10, 10, 0.2)

    # Initialize
    agents = []
    for i in range(2, 9):
        start, goal = w.get_start_goal(0.5)
        agents.append(Agent(i, start, goal))

    global_path = dict()
    for agent in agents:
        print("=========")
        print(agent)
        astar = AStar(agent, w, global_path)
        path = astar.search()
        global_path[agent.id] = path
        print("Conflicts found: %d" % agent.conflicts_found)
        print("Path:")
        for p in path:
            print(p)
    dv = DataViz(w, agents)
    dv.plot_paths(global_path)
