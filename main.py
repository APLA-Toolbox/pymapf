from src.cooperative_astar import CooperativeAStar
from src.world import World


if __name__ == "__main__":
    # all coordinates are given in format [y,x]
    # import doctest

    # doctest.testmod()

    w = World(15, 15, 0.1)

    # Initialize
    cas = CooperativeAStar(w)
    start, goal = w.get_start_goal(0.6)
    cas.register_agent("A", start, goal)
    start, goal = w.get_start_goal(0.6)
    cas.register_agent("B", start, goal)
    start, goal = w.get_start_goal(0.3)
    cas.register_agent("C", start, goal)

    cas.run_simulation()
    cas.visualize("test")
