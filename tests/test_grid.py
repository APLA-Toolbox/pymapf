import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

from pymapf.centralized.world import World
from pymapf.core.grid import GridMap


def test_dimensions_and_free_cells():
    g = GridMap([[0, 0, 0], [0, 1, 0]])
    assert g.height == 2 and g.width == 3
    assert g.free_cells == 5  # one blocked


def test_is_free_and_bounds():
    g = GridMap([[0, 1], [0, 0]])
    assert g.is_free((0, 0))
    assert not g.is_free((0, 1))  # blocked
    assert not g.is_free((-1, 0))  # out of bounds
    assert not g.is_free((2, 0))


def test_neighbors_orthogonal():
    g = GridMap([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert set(g.neighbors((1, 1))) == {(0, 1), (2, 1), (1, 0), (1, 2)}


def test_neighbors_skip_blocked():
    g = GridMap([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    assert (0, 1) not in g.neighbors((1, 1))


def test_neighbors_diagonal_requires_open_corner():
    # Diagonal (0,0)->(1,1) is blocked because both corners (0,1),(1,0) are walls.
    g = GridMap([[0, 1], [1, 0]])
    assert (1, 1) not in g.neighbors((0, 0), allow_diagonals=True)
    # With open corners the diagonal is allowed.
    g2 = GridMap([[0, 0], [0, 0]])
    assert (1, 1) in g2.neighbors((0, 0), allow_diagonals=True)


def test_invalid_grid():
    with pytest.raises(ValueError):
        GridMap([])
    with pytest.raises(ValueError):
        GridMap([[0, 0], [0]])  # ragged


def test_from_world():
    w = World(6, 5, 0)  # no random walls, only the border
    g = GridMap.from_world(w)
    assert g.height == w.height and g.width == w.length
    assert not g.is_free((0, 0))  # border is a wall
    assert g.is_free((2, 2))  # interior is free
