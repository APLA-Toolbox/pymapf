import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pymapf.algorithms.space_time_astar import space_time_astar
from pymapf.core.grid import GridMap
from pymapf.core.solver import Constraints


def test_straight_line():
    g = GridMap([[0, 0, 0, 0, 0]])
    p = space_time_astar(g, (0, 0), (0, 4))
    assert p == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


def test_no_path_when_walled_off():
    g = GridMap([[0, 1, 0]])
    assert space_time_astar(g, (0, 0), (0, 2)) is None


def test_vertex_constraint_forces_wait():
    g = GridMap([[0, 0, 0]])
    c = Constraints()
    c.add_vertex((0, 1), 1)  # can't be at middle cell at t=1
    p = space_time_astar(g, (0, 0), (0, 2), constraints=c)
    # Must wait one step, so it reaches goal at t=3 instead of t=2.
    assert p[-1] == (0, 2)
    assert len(p) - 1 == 3
    assert ((0, 1), 1) not in [(cell, t) for t, cell in enumerate(p)]


def test_goal_constraint_delays_settling():
    g = GridMap([[0, 0, 0]])
    c = Constraints()
    c.add_vertex((0, 2), 2)  # goal occupied at t=2
    p = space_time_astar(g, (0, 0), (0, 2), constraints=c)
    # Cannot settle on goal until after t=2.
    arrival = len(p) - 1
    assert p[-1] == (0, 2)
    assert arrival > 2


def test_edge_constraint_blocks_move():
    g = GridMap([[0, 0]])
    c = Constraints()
    c.add_edge((0, 0), (0, 1), 1)  # cannot move (0,0)->(0,1) arriving t=1
    p = space_time_astar(g, (0, 0), (0, 1), constraints=c)
    # Only route is that edge, but not at t=1; it must wait then move at t=2.
    assert p == [(0, 0), (0, 0), (0, 1)]
