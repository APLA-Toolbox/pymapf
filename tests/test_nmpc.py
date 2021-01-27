# -*- coding: utf-8 -*-

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pymapf.decentralized.nmpc import MultiAgentNMPC
from pymapf.decentralized.position import Position


def test_nmpc_agents():
    nmpc = MultiAgentNMPC()
    nmpc.register_agent("toto", Position(1, 2), Position(4, 5))
    nmpc.register_agent("tata", Position(2, 2), Position(7, 5))
    nmpc.register_agent("titi", Position(2, 3), Position(4, 8))
    assert len(nmpc.agents) == 3


def test_nmpc_obstacles():
    nmpc = MultiAgentNMPC()
    nmpc.register_obstacle(2, 3.14, Position(2, 3))
    assert len(nmpc.obstacles_objects) == 1

def test_sim_no_obstacles():
    nmpc = MultiAgentNMPC()
    nmpc.register_agent("toto", Position(1, 2), Position(4, 5))
    nmpc.register_agent("tata", Position(2, 2), Position(7, 5))
    nmpc.register_agent("titi", Position(2, 3), Position(4, 8))
    nmpc.run_simulation()

def test_sim_obstacles():
    nmpc = MultiAgentNMPC()
    nmpc.register_agent("toto", Position(1, 2), Position(4, 5))
    nmpc.register_agent("tata", Position(2, 2), Position(7, 5))
    nmpc.register_agent("titi", Position(2, 3), Position(4, 8))
    nmpc.register_obstacle(2, 3.14, Position(2, 3))
    nmpc.register_obstacle(2, -3.14, Position(10, 7))
    nmpc.run_simulation()

def test_visualize_no_obs():
    nmpc = MultiAgentNMPC()
    nmpc.register_agent("toto", Position(1, 2), Position(4, 5))
    nmpc.register_agent("tata", Position(2, 2), Position(7, 5))
    nmpc.register_agent("titi", Position(2, 3), Position(4, 8))
    nmpc.run_simulation()
    nmpc.visualize("toto", 10, 10)

def test_visualize_obs():
    nmpc = MultiAgentNMPC()
    nmpc.register_agent("toto", Position(1, 2), Position(4, 5))
    nmpc.register_agent("tata", Position(2, 2), Position(7, 5))
    nmpc.register_agent("titi", Position(2, 3), Position(4, 8))
    nmpc.register_obstacle(2, 3.14, Position(2, 3))
    nmpc.register_obstacle(2, -3.14, Position(10, 7))
    nmpc.run_simulation()
    nmpc.visualize("toto", 10, 10)

