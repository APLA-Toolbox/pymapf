from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pymapf.centralized.cooperative_astar.agent import Agent
from pymapf.centralized.world import World
from pymapf.decentralized.position import Position
import numpy as np
import random
import time


def test_agent():
    a = Agent("h", (2, 2), (3, 3), allow_diagonals=True)
    assert a.init_pos == (2, 2) and a.goal_pos == (3, 3)
