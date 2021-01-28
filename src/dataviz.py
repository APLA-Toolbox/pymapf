import os
import matplotlib as mpl
from random import random

if "DISPLAY" not in os.environ:
    mpl.use("agg")
else:
    mpl.use("TkAgg")
from matplotlib import pyplot as plt

plt.style.use("ggplot")


class DataViz:
    def __init__(self, world, agents):
        ox, oy = [], []
        for y in range(world.length):
            for x in range(world.height):
                if world.grid[x][y] == 1:
                    ox.append(x)
                    oy.append(y)

        plt.plot(ox, oy, "ok")

        self.agents_colors = dict()
        for agent in agents:
            self.agents_colors[agent.id] = [random(), random(), random()]
            plt.plot(
                agent.init_pos[0],
                agent.init_pos[1],
                marker="o",
                color=self.agents_colors[agent.id],
            )
            plt.plot(
                agent.goal_pos[0],
                agent.goal_pos[1],
                marker="x",
                color=self.agents_colors[agent.id],
            )

        plt.grid(True)

    def plot_paths(self, paths):
        agents = list(paths.keys())

        i = 1
        it = dict()
        for a in agents:
            it[a] = i
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        while True:
            for a in agents:
                if it[a] > len(paths[a]) - 1:
                    del paths[a]
                    agents.remove(a)
                    continue
                x = [paths[a][it[a]].x, paths[a][it[a] - 1].x]
                y = [paths[a][it[a]].y, paths[a][it[a] - 1].y]
                plt.plot(y, x, marker="o", color=self.agents_colors[a])
                plt.pause(0.5)
                it[a] += 1
            if len(agents) == 0:
                break
        plt.show()
