from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pymapf.decentralized.nmpc import MultiAgentNMPC
from pymapf.decentralized.position import Position
import numpy as np
import time

runtimes = []

sim = MultiAgentNMPC()
sim.register_agent("1", Position(0, 5), Position(10, 5), vmin=0)
sim.register_agent("2", Position(10, 5), Position(0, 5), vmin=0)
stamp = time.time()
sim.run_simulation()
runtimes.append((time.time() - stamp, 2))

sim = MultiAgentNMPC()
sim.register_agent("1", Position(0, 5), Position(10, 5), vmin=0)
sim.register_agent("2", Position(10, 5), Position(0, 5), vmin=0)
sim.register_agent("3", Position(5, 10), Position(5, 0), vmin=0)
sim.register_agent("4", Position(5, 0), Position(5, 10), vmin=0)
stamp = time.time()
sim.run_simulation()
runtimes.append((time.time() - stamp, 4))

sim = MultiAgentNMPC()
sim.register_agent("1", Position(0, 5), Position(10, 5), vmin=0)
sim.register_agent("2", Position(10, 5), Position(0, 5), vmin=0)
sim.register_agent("3", Position(5, 10), Position(5, 0), vmin=0)
sim.register_agent("4", Position(5, 0), Position(5, 10), vmin=0)
sim.register_agent("5", Position(0, 0), Position(10, 10), vmin=0)
sim.register_agent("6", Position(10, 10), Position(0, 0), vmin=0)
sim.register_agent("7", Position(10, 0), Position(0, 10), vmin=0)
sim.register_agent("8", Position(0, 10), Position(10, 0), vmin=0)
stamp = time.time()
sim.run_simulation()
runtimes.append((time.time() - stamp, 8))

sim = MultiAgentNMPC(simulation_time=16)
sim.register_agent("1", Position(0, 5), Position(20, 15), vmin=0)
sim.register_agent("2", Position(10, 5), Position(10, 15), vmin=0)
sim.register_agent("3", Position(5, 10), Position(15, 10), vmin=0)
sim.register_agent("4", Position(5, 0), Position(15, 20), vmin=0)
sim.register_agent("5", Position(0, 0), Position(20, 20), vmin=0)
sim.register_agent("6", Position(10, 10), Position(10, 10), vmin=0)
sim.register_agent("7", Position(10, 0), Position(10, 20), vmin=0)
sim.register_agent("8", Position(0, 10), Position(20, 10), vmin=0)

sim.register_agent("9", Position(10, 15), Position(10, 5), vmin=0)
sim.register_agent("10", Position(20, 15), Position(0, 5), vmin=0)
sim.register_agent("11", Position(15, 20), Position(5, 0), vmin=0)
sim.register_agent("12", Position(15, 10), Position(5, 10), vmin=0)
sim.register_agent("13", Position(20, 20), Position(0, 0), vmin=0)
sim.register_agent("14", Position(20, 10), Position(0, 10), vmin=0)
sim.register_agent("15", Position(10, 20), Position(10, 0), vmin=0)

sim.register_agent("16", Position(20, 5), Position(0, 15), vmin=0)
sim.register_agent("17", Position(15, 0), Position(5, 20), vmin=0)
sim.register_agent("18", Position(20, 0), Position(0, 20), vmin=0)

sim.register_agent("19", Position(0, 15), Position(20, 5), vmin=0)
sim.register_agent("20", Position(5, 20), Position(15, 0), vmin=0)
sim.register_agent("21", Position(0, 20), Position(20, 0), vmin=0)
stamp = time.time()
sim.run_simulation()
runtimes.append((time.time() - stamp, 32))

