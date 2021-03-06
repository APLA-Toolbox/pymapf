"""
Decentralized planning using reactive control (nonlinear model predictive control)
author: Erwin Lejeune (erwin.lejeune15@gmail.com)
"""

from ..obstacle import Obstacle
from .nmpc_agent import NMPCAgent
import numpy as np
import matplotlib as mpl
from random import random
import os

if "DISPLAY" not in os.environ:
    mpl.use("agg")
else:
    mpl.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import coloredlogs

# import threading
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)


class MultiAgentNMPC:
    def __init__(
        self, simulation_time=8.0, timestep=0.1, nmpc_timestep=0.3, log_level="WARNING"
    ):
        self.simulation_time = simulation_time
        self.timestep = timestep
        self.number_of_timesteps = int(simulation_time / timestep)
        self.nmpc_timestep = nmpc_timestep

        # Agents
        self.agents = dict()
        self.global_state_history = dict()

        # Obstacles
        self.obstacles_objects = []

        # Flags
        self.simulation_complete = False
        self.__init_logger(log_level)
        coloredlogs.install(level=log_level)

    def register_agent(
        self,
        ident,
        start,
        goal,
        qc=5.0,
        kappa=4.0,
        radius=0.5,
        vmax=2,
        vmin=0.2,
        horizon_length=4,
    ):
        if ident in self.agents:
            logging.warning("Failed to register agent: ID has already been registered.")
            return
        agent = NMPCAgent(
            ident,
            start,
            goal,
            self.number_of_timesteps,
            self.nmpc_timestep,
            self.timestep,
            qc,
            kappa,
            radius,
            vmax,
            vmin,
            horizon_length,
        )
        self.agents[ident] = agent

    def register_obstacle(self, velocity, theta, initial_position):
        obstacle = Obstacle(
            False,
            velocity,
            theta,
            initial_position,
            self.simulation_time,
            self.number_of_timesteps,
        )
        self.obstacles_objects.append(obstacle)
        if not hasattr(self, "obstacles"):
            self.obstacles = obstacle.state
        else:
            self.obstacles = np.dstack((self.obstacles, obstacle.state))

    def run_simulation(self):
        try:
            obstacles = self.obstacles
        except BaseException as e:
            logging.debug(e)
            obstacles = []
        for i in range(self.number_of_timesteps):
            # self.other_agents = dict()
            other_agents = []
            # threads = []
            for key, agent in self.agents.items():
                other_agents = self.__agent_step(key, agent, i, obstacles, other_agents)
            #     threads.append(
            #         threading.Thread(
            #             target=self.__agent_step,
            #             args=(key, agent, i, obstacles),
            #         )
            #     )
            #     threads[-1].start()
            # for t in threads:
            #     t.join()
        self.simulation_complete = True

    def visualize(self, saved_file, map_length, map_height):
        if not self.simulation_complete:
            logging.warning("Failed to visualize: simulation must be complete")
            return

        self.__plot(saved_file, map_length, map_height)

    def __agent_step(self, key, agent, i, obstacles, other_agents_lst):
        # other_agents = self.other_agents.copy()
        # try:
        #     del other_agents[key]
        # except BaseException as e:
        #     logging.debug(e)
        #     pass
        # other_agents_lst = list(other_agents.values())
        state_history, vel, state = agent.simulate_step(i, obstacles, other_agents_lst)
        agent_as_obstacle = self.__agent_to_obstacle(vel, state)
        # self.other_agents[key] = agent_as_obstacle
        other_agents_lst.append(agent_as_obstacle)
        self.global_state_history[key] = state_history
        return other_agents_lst

    def __agent_to_obstacle(self, velocity, pos):
        return np.concatenate((pos, velocity))

    def __init_logger(self, log_level):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        logging.basicConfig(
            filename="logs/main.log",
            format="%(levelname)s:%(message)s",
            filemode="w",
            level=log_level,
        )

    def __plot(self, saved_file, map_length, map_height):
        fig = plt.figure()
        ax = fig.add_subplot(
            111, autoscale_on=False, xlim=(0, map_length), ylim=(0, map_height)
        )
        ax.set_aspect("equal")
        ax.grid()

        lines = []
        agents_list = []
        for key, agent in self.agents.items():
            rgb = [random(), random(), random()]
            agents_list.append(
                Circle(
                    (
                        self.global_state_history[key][0, 0],
                        self.global_state_history[key][0, 1],
                    ),
                    agent.radius,
                    facecolor=rgb,
                    edgecolor="black",
                )
            )
            (line,) = ax.plot([], [], c=rgb)
            lines.append(line)
        obstacle_list = []
        try:
            for obstacle in range(np.shape(self.obstacles)[2]):
                obstacle_list.append(
                    Circle(
                        (0, 0),
                        self.obstacles_objects[obstacle].radius,
                        facecolor="red",
                        edgecolor="black",
                    )
                )
        except BaseException as e:
            logging.debug(e)

        def init():
            for agent in agents_list:
                ax.add_patch(agent)
                line.set_data([], [])
                lines.append(line)
            for obstacle in obstacle_list:
                ax.add_patch(obstacle)

            return agents_list + lines + obstacle_list

        def animate(i):
            k = 0
            for key, _ in self.agents.items():
                agents_list[k].center = (
                    self.global_state_history[key][0, i],
                    self.global_state_history[key][1, i],
                )
                lines[k].set_data(
                    self.global_state_history[key][0, :i],
                    self.global_state_history[key][1, :i],
                )
                k += 1
            for j, _ in enumerate(obstacle_list):
                obstacle_list[j].center = (
                    self.obstacles[0, i, j],
                    self.obstacles[1, i, j],
                )
            return agents_list + lines + obstacle_list

        init()
        step = self.simulation_time / self.number_of_timesteps
        for i in range(self.number_of_timesteps):
            animate(i)
            plt.pause(step)

        # Save animation
        if not saved_file:
            return

        ani = animation.FuncAnimation(
            fig,
            animate,
            np.arange(1, self.number_of_timesteps),
            interval=200,
            blit=True,
            init_func=init,
        )

        ani.save(saved_file + str(".gif"), writer="ffmpeg", fps=10)
