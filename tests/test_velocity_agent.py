# -*- coding: utf-8 -*-

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pymapf.decentralized.velocity_agent import VelocityAgent
from pymapf.decentralized.position import Position
import numpy as np


def test_nmpc_agent():
    agent = VelocityAgent("toto", Position(2, 2), Position(4, 6), 15, 0.1)
    assert agent.current_state.all() == np.array([2, 2]).all()
    assert agent.vmax > agent.vmin
