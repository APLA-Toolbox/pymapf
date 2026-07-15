import sys
from math import sqrt
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

from pymapf.core import heuristics as H
from pymapf.core.heuristics import get_heuristic


def test_manhattan():
    assert H.manhattan((0, 0), (3, 4)) == 7.0


def test_euclidean():
    assert H.euclidean((0, 0), (3, 4)) == 5.0


def test_chebyshev():
    assert H.chebyshev((0, 0), (3, 4)) == 4.0


def test_octile():
    assert H.octile((0, 0), (0, 4)) == pytest.approx(4.0)
    assert H.octile((0, 0), (3, 3)) == pytest.approx(3 * sqrt(2))


def test_get_heuristic_by_name_and_callable():
    assert get_heuristic("manhattan") is H.manhattan
    custom = lambda a, b: 0.0
    assert get_heuristic(custom) is custom


def test_get_heuristic_unknown():
    with pytest.raises(ValueError):
        get_heuristic("does-not-exist")
