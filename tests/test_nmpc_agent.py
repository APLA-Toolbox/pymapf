from pymapf.decentralized.nmpc_agent import NMPCAgent
from pymapf.decentralized.position import Position
import numpy as np

def test_nmpc_agent():
    agent = NMPCAgent("toto", Position(2, 2), Position(4, 10), 15, .3, .1)
    assert agent.current_state == np.array([2, 2])
    assert agent.vmax > agent.vmin

