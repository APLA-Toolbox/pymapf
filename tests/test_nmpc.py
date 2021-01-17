from pymapf.decentralized.nmpc import MultiAgentNMPC

ag = MultiAgentNMPC()

def test_agent_creation():
    ag.register_agent("rbt", np.array([0, 0]), np.array([5, 5]))
    assert len(ag.agents) > 0

def test_obstacle_creation():
    ag.register_obstacle(2, 0, np.array([0, 5]))
    assert len(ag.obstacles_objects) > 0

def test_simulation():
    ag.run_simulation()
    assert ag.simulation_complete == True
