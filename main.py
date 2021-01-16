from pymapf.decentralized.nmpc import MultiAgentNMPC
from pymapf.decentralized.velocity_obstacle import MultiAgentVelocityObstacle
import numpy as np

def main_nmpc():
    sim = MultiAgentNMPC()
    sim.register_agent("r2d2", np.array([5, 5]), np.array([5, 5]))
    sim.register_agent("bb8", np.array([5, 0]), np.array([5, 10]))
    sim.register_obstacle(-2, np.pi/2, np.array([5, 12]))
    sim.register_obstacle(2, 0, np.array([0, 5]))
    # sim.register_obstacle(2, -np.pi * 3 / 4, np.array([10, 10]))
    # sim.register_obstacle(2, np.pi * 3 / 4, np.array([7.5, 2.5]))

    sim.run_simulation()
    sim.visualize("nmpc_scenario")

def main_vel_obstacle():
    sim = MultiAgentVelocityObstacle()
    sim.register_agent("r2d2", np.array([5, 5]), np.array([5, 5]))
    sim.register_agent("bb8", np.array([5, 0]), np.array([5, 10]))
    sim.register_obstacle(-2, np.pi/2, np.array([5, 12]))
    sim.register_obstacle(2, 0, np.array([0, 5]))
    # sim.register_obstacle(2, -np.pi * 3 / 4, np.array([10, 10]))
    # sim.register_obstacle(2, np.pi * 3 / 4, np.array([7.5, 2.5]))

    sim.run_simulation()
    sim.visualize("vel_scenario")

if __name__ == "__main__":
    main_vel_obstacle()
    main_nmpc()
