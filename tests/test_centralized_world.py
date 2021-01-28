from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pymapf.centralized.world import World
import numpy as np
import random
import time

def test_world_plot():
    w = World(10, 10, 0)
    w.plot_grid()
    