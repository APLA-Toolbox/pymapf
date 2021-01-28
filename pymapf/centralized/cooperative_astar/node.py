from math import sqrt
from ..common import HEURISTIC
from .state import State


class Node:
    """
    >>> k = Node(0, 0, 4, 3, 0, None)
    >>> k.calculate_heuristic()
    5.0
    >>> n = Node(1, 4, 3, 4, 2, None)
    >>> n.calculate_heuristic()
    2.0
    >>> l = [k, n]
    >>> n == l[0]
    False
    >>> l.sort()
    >>> n == l[0]
    True
    """

    def __init__(
        self,
        pos_x: int,
        pos_y: int,
        t: int,
        goal_x: int,
        goal_y: int,
        g_cost: float,
        is_closed: bool,
        is_open: bool,
        parent,
    ):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.t = t
        self.state = State(pos_x, pos_y, t)
        self.pos = (pos_y, pos_x)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.is_closed = is_closed
        self.is_open = is_open
        self.g_cost = g_cost
        self.parent = parent
        self.h_cost = self.calculate_heuristic()
        self.f_cost = self.g_cost + self.h_cost

    def calculate_heuristic(self) -> float:
        """
        Heuristic for the A*
        """
        dy = self.pos_x - self.goal_x
        dx = self.pos_y - self.goal_y
        if HEURISTIC == 1:
            return abs(dx) + abs(dy)
        return sqrt(dy ** 2 + dx ** 2)

    def print(self):
        print("Node : <State = [x=%d; y=%d; t=%d]>" % (self.pos_x, self.pos_y, self.t))

    def __str__(self):
        return "Node : <State = [x=%d; y=%d; t=%d]>" % (self.pos_x, self.pos_y, self.t)

    def __lt__(self, other) -> bool:
        return self.f_cost <= other.f_cost
