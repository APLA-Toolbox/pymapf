from .common import DIAGONALS
import numpy as np
from math import sqrt
from typing import List, Tuple
from sys import stdout
import random
from termcolor import colored
from .node import Node


class World:
    def __init__(self, length: int, height: int, p_walls: float):
        self.length = length
        self.height = height
        self.p_walls = p_walls
        self.path_added = False

        self.grid = np.zeros((height, length), int)
        self.__generate_walls()

        if not DIAGONALS:
            # up, left, down, right
            self.delta = [[-1, 0, 1], [0, -1, 1], [1, 0, 1], [0, 1, 1]]
        else:
            # up, left, down, right
            # upleft, upright, downleft, downright
            self.delta = [
                [-1, 0, 1],
                [0, -1, 1],
                [1, 0, 1],
                [0, 1, 1],
                [-1, -1, sqrt(2)],
                [-1, 1, sqrt(2)],
                [1, -1, sqrt(2)],
                [1, 1, sqrt(2)],
            ]

    def add_path(self, path: List[Tuple[int]]):
        i = 0
        for elem in path:
            if i == 0 or i == len(path) - 1:
                self.grid[elem[0]][elem[1]] = 3
            else:
                self.grid[elem[0]][elem[1]] = 2
            i += 1
        self.path_added = True

    def plot_grid(self):
        for rows in self.grid:
            for elem in rows:
                if elem == 1:
                    stdout.write(colored("█", "red"))
                elif elem == 0:
                    stdout.write(" ")
                elif elem == 2:
                    stdout.write(colored("¤", "blue"))
                elif elem == 3:
                    stdout.write(colored("x", "green"))
                elif elem == 4:
                    stdout.write(colored("*", "blue"))
                elif elem == 5:
                    stdout.write(colored("~", "magenta"))
                elif elem == 6:
                    stdout.write(colored("+", "cyan"))
                else:
                    stdout.write("o", "yellow")
            print()

    def get_random_available_position(self) -> Tuple[int]:
        i = 0
        while 1:
            i += 1
            random_row = random.randint(0, self.height - 1)
            random_col = random.randint(0, self.length - 1)

            if self.grid[random_row][random_col] != 1:
                break
            if i > self.length * self.height * 100:
                print(colored("Couldn't find random available position.", "red"))
                return None

        return (random_row, random_col)

    def get_start_goal(self, pHeuristic) -> Tuple[Tuple[int]]:
        i = 0

        # Initialization
        start = self.get_random_available_position()
        goal = self.get_random_available_position()
        n = Node(start[1], start[0], 0, goal[1], goal[0], 0, False, False, None)
        h = n.calculate_heuristic()

        # Set goal value
        nTarget = Node(
            0, 0, -1, self.length - 1, self.height - 1, 0, False, False, None
        )
        hTarget = nTarget.calculate_heuristic()

        while h < hTarget * pHeuristic:
            i += 1
            start = self.get_random_available_position()
            goal = self.get_random_available_position()
            n = Node(start[1], start[0], 0, goal[1], goal[0], 0, False, False, None)
            h = n.calculate_heuristic()
            if i > 100 * self.length * self.height:
                print(
                    colored(
                        "Couldn't find start and goal respecting the constraints.",
                        "red",
                    )
                )
                return None
        return start, goal

    def change_grid(self, position: Tuple[int], value: int):
        self.grid[position[1]][position[0]] = value

    def __generate_walls(self):
        for x in range(self.height):
            for y in range(self.length):
                if random.random() < self.p_walls:
                    self.grid[x][y] = 1
                if x == 0 or x == self.height - 1:
                    self.grid[x][y] = 1
                if y == 0 or y == self.length - 1:
                    self.grid[x][y] = 1
