from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pymapf.decentralized.nmpc import MultiAgentNMPC
from pymapf.decentralized.position import Position
import numpy as np

sim = MultiAgentNMPC()
sim.register_agent("r2d2", Position(0, 3), Position(10, 7), vmin=0)
sim.register_agent("bb8", Position(0, 7), Position(5, 10), vmin=0)
sim.register_agent("c3po", Position(10, 7), Position(5, 0), vmin=0)
sim.register_agent("r4d4", Position(10, 3), Position(0, 3), vmin=0)
sim.register_agent("wally", Position(5, 10), Position(0, 7), vmin=0)
sim.register_agent("spot", Position(5, 0), Position(10, 3), vmin=0)
sim.run_simulation()
sim.visualize("switch_positions_nmpc", 10, 10)
