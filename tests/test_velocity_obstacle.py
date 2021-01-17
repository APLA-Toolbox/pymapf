# -*- coding: utf-8 -*-

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pymapf.decentralized.velocity_obstacle import MultiAgentVelocityObstacle
from pymapf.decentralized.position import Position

def test_mavo_agents():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_agent("toto", Position(2, 2), Position(4, 5))
    mavo.register_agent("tata", Position(2, 2), Position(4, 5))
    mavo.register_agent("titi", Position(2, 2), Position(4, 5))
    assert(len(mavo.agents) == 3)

def test_mavo_obstacles():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_obstacle(2, 3.14, Position(2, 3))
    assert(len(mavo.obstacles_objects) == 1)

