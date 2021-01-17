from pymapf.decentralized.velocity_agent import VelocityAgent
from pymapf.decentralized.position import Position
import numpy as np

def test_nmpc_agent():
    agent = VelocityAgent("toto", Position(2, 2), Position(4, 6), 15, .1)
    assert agent.current_state == np.array([2, 2])
    assert agent.vmax > agent.vmin

