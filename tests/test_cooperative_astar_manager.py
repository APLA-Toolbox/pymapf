from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pymapf.centralized.cooperative_astar.cooperative_astar import CooperativeAStar
from pymapf.centralized.world import World
from pymapf.decentralized.position import Position
import numpy as np
import random
import time


def test_cooperative_astar_sim():
    w = World(12, 12, 0.1)
    campf = CooperativeAStar(w)
    agents_labels = ["A", "B", "C", "D", "E"]
    for label in agents_labels:
        start, goal = w.get_start_goal(random.random() * 0.5)
        campf.register_agent(label, start, goal)

    campf.run_simulation()


def test_cooperative_astar_sim_diagonals():
    w = World(12, 12, 0.1, allow_diagonals=True)
    campf = CooperativeAStar(w)
    agents_labels = ["A", "B", "C", "D", "E"]
    for label in agents_labels:
        start, goal = w.get_start_goal(random.random() * 0.5)
        campf.register_agent(label, start, goal)

    campf.run_simulation()
    campf.visualize("test")


def test_cooperative_astar_viz():
    w = World(12, 12, 0.1)
    campf = CooperativeAStar(w)
    agents_labels = ["A", "B", "C", "D", "E"]
    for label in agents_labels:
        start, goal = w.get_start_goal(random.random() * 0.5)
        campf.register_agent(label, start, goal)

    campf.run_simulation()
    campf.visualize("test")
