import numpy as np


class Obstacle:
    def __init__(
        self,
        is_agent,
        velocity,
        theta,
        initial_position,
        simulation_time,
        number_of_timesteps,
        radius=0.5,
    ):
        self.is_agent = is_agent
        self.velocity = velocity
        self.initial_position = np.array([initial_position.x, initial_position.y])
        self.theta = theta
        self.radius = radius

        # Creates an obstacle starting at initial position and moving at velocity in theta direction
        t = np.linspace(0, simulation_time, number_of_timesteps)
        theta = self.theta * np.ones(np.shape(t))
        vx = self.velocity * np.cos(theta)
        vy = self.velocity * np.sin(theta)
        v = np.stack([vx, vy])
        p0 = self.initial_position.reshape((2, 1))
        p = p0 + np.cumsum(v, axis=1) * (simulation_time / number_of_timesteps)
        self.state = np.concatenate((p, v)).reshape(4, number_of_timesteps, 1)

    def __str__(self):
        return str(self.state)
