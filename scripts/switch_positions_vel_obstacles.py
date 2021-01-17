from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pymapf.decentralized.velocity_obstacle import MultiAgentVelocityObstacle
from pymapf.decentralized.position import Position
import numpy as np

sim = MultiAgentVelocityObstacle(simulation_time=8.0)
sim.register_agent("r2d2", Position(0, 3), Position(10, 7))
sim.register_agent("bb8", Position(0, 7), Position(5, 10))
sim.register_agent("c3po", Position(10, 7), Position(5, 0))
sim.register_agent("r4d4", Position(10, 3), Position(0, 3))
sim.register_agent("wally", Position(5, 10), Position(0, 7))
sim.register_agent("spot", Position(5, 0), Position(10, 3))
sim.run_simulation()
sim.visualize("switch_positions_velocity_obstacles", 10, 10)
