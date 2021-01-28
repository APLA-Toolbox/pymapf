# -*- coding: utf-8 -*-

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pymapf.decentralized.velocity_obstacle.velocity_obstacle import (
    MultiAgentVelocityObstacle,
)
from pymapf.decentralized.position import Position


def test_mavo_agents():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_agent("toto", Position(2, 2), Position(10, 5))
    mavo.register_agent("tata", Position(2, 5), Position(1, 5))
    mavo.register_agent("titi", Position(2, 9), Position(8, 5))
    assert len(mavo.agents) == 3


def test_mavo_obstacles():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_obstacle(2, 3.14, Position(2, 3))
    assert len(mavo.obstacles_objects) == 1


def test_mavo_simulation_no_obs():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_agent("toto", Position(2, 5), Position(4, 10))
    mavo.register_agent("tata", Position(4, 2), Position(4, 1))
    mavo.register_agent("titi", Position(10, 2), Position(0, 5))
    mavo.run_simulation()


def test_mavo_simulation_obstacles():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_agent("tata", Position(2, 2), Position(4, 5))
    mavo.register_obstacle(2, 3.14, Position(2, 3))
    mavo.register_obstacle(2, -3.14, Position(2, 10))
    mavo.run_simulation()


def test_mavo_no_obs_vis():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_agent("toto", Position(2, 5), Position(4, 10))
    mavo.register_agent("tata", Position(4, 2), Position(4, 1))
    mavo.register_agent("titi", Position(10, 2), Position(0, 5))
    mavo.run_simulation()
    mavo.visualize("tata", 10, 10)


def test_mavo_obs_vis():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_agent("tata", Position(2, 2), Position(4, 5))
    mavo.register_obstacle(2, 3.14, Position(2, 3))
    mavo.register_obstacle(2, -3.14, Position(2, 10))
    mavo.run_simulation()
    mavo.visualize("toto", 10, 10)
