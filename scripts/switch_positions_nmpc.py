from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pymapf.decentralized.nmpc import MultiAgentNMPC
from pymapf.decentralized.position import Position
import numpy as np
import time
import logging

start = time.time()
sim = MultiAgentNMPC()
sim.register_agent("r2d2", Position(0, 3), Position(10, 7), vmin=0)
sim.register_agent("bb8", Position(0, 7), Position(5, 10), vmin=0)
sim.register_agent("c3po", Position(10, 7), Position(5, 0), vmin=0)
sim.register_agent("r4d4", Position(10, 3), Position(0, 3), vmin=0)
sim.register_agent("wally", Position(5, 10), Position(0, 7), vmin=0)
sim.register_agent("spot", Position(5, 0), Position(10, 3), vmin=0)
sim.run_simulation()
t = time.time() - start
logging.warning("Time for simulation (single thread): %.2f secs" % t)

start = time.time()
sim_threaded = MultiAgentNMPC(threaded_mode=True)
sim_threaded.register_agent("r2d2", Position(0, 3), Position(10, 7), vmin=0)
sim_threaded.register_agent("bb8", Position(0, 7), Position(5, 10), vmin=0)
sim_threaded.register_agent("c3po", Position(10, 7), Position(5, 0), vmin=0)
sim_threaded.register_agent("r4d4", Position(10, 3), Position(0, 3), vmin=0)
sim_threaded.register_agent("wally", Position(5, 10), Position(0, 7), vmin=0)
sim_threaded.register_agent("spot", Position(5, 0), Position(10, 3), vmin=0)
sim_threaded.run_simulation()
t = time.time() - start
logging.warning("Time for simulation (multiprocessing): %.2f secs" % t)


sim_threaded.visualize("switch_positions_nmpc_threaded", 12, 12)
sim.visualize("switch_positions_nmpc", 12, 12)
