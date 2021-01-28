from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pymapf.decentralized.nmpc.nmpc import MultiAgentNMPC
from pymapf.decentralized.position import Position
import numpy as np
import time

runtimes = []
global_runtimes_per_agents = []

sim = MultiAgentNMPC()
sim.register_agent("1", Position(0, 5), Position(10, 5), vmin=0)
sim.register_agent("2", Position(10, 5), Position(0, 5), vmin=0)
sim.register_agent("3", Position(5, 10), Position(5, 0), vmin=0)
sim.register_agent("4", Position(5, 0), Position(5, 10), vmin=0)
stamp = time.time()
sim.run_simulation()
runtimes_per_agents = {}
for key, agent in sim.agents.items():
    runtimes_per_agents[key] = agent.total_computation_runtime
global_runtimes_per_agents.append(runtimes_per_agents)
runtimes.append((time.time() - stamp, 100))

sim = MultiAgentNMPC()
sim.register_agent("1", Position(0, 6), Position(12, 6), vmin=0)
sim.register_agent("2", Position(12, 6), Position(0, 6), vmin=0)
sim.register_agent("3", Position(6, 12), Position(6, 0), vmin=0)
sim.register_agent("4", Position(6, 0), Position(6, 12), vmin=0)
stamp = time.time()
sim.run_simulation()
runtimes_per_agents = {}
for key, agent in sim.agents.items():
    runtimes_per_agents[key] = agent.total_computation_runtime
global_runtimes_per_agents.append(runtimes_per_agents)
runtimes.append((time.time() - stamp, 144))

sim = MultiAgentNMPC()
sim.register_agent("1", Position(0, 8), Position(16, 8), vmin=0)
sim.register_agent("2", Position(16, 8), Position(0, 8), vmin=0)
sim.register_agent("3", Position(8, 16), Position(8, 0), vmin=0)
sim.register_agent("4", Position(8, 0), Position(8, 16), vmin=0)
stamp = time.time()
sim.run_simulation()
runtimes_per_agents = {}
for key, agent in sim.agents.items():
    runtimes_per_agents[key] = agent.total_computation_runtime
global_runtimes_per_agents.append(runtimes_per_agents)
runtimes.append((time.time() - stamp, 256))

sim = MultiAgentNMPC()
sim.register_agent("1", Position(0, 10), Position(20, 10), vmin=0)
sim.register_agent("2", Position(20, 10), Position(0, 10), vmin=0)
sim.register_agent("3", Position(10, 20), Position(10, 0), vmin=0)
sim.register_agent("4", Position(10, 0), Position(10, 20), vmin=0)
stamp = time.time()
sim.run_simulation()
runtimes_per_agents = {}
for key, agent in sim.agents.items():
    runtimes_per_agents[key] = agent.total_computation_runtime
global_runtimes_per_agents.append(runtimes_per_agents)
runtimes.append((time.time() - stamp, 400))

sim = MultiAgentNMPC()
sim.register_agent("1", Position(0, 25), Position(50, 25), vmin=0)
sim.register_agent("2", Position(50, 25), Position(0, 25), vmin=0)
sim.register_agent("3", Position(25, 50), Position(25, 0), vmin=0)
sim.register_agent("4", Position(25, 0), Position(25, 50), vmin=0)
stamp = time.time()
sim.run_simulation()
runtimes_per_agents = {}
for key, agent in sim.agents.items():
    runtimes_per_agents[key] = agent.total_computation_runtime
global_runtimes_per_agents.append(runtimes_per_agents)
runtimes.append((time.time() - stamp, 2500))

print(runtimes)
print(global_runtimes_per_agents)
