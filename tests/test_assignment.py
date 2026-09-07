"""The Hungarian algorithm against brute force."""

import itertools
import math
import random
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest  # noqa: E402

from pymapf.core.assignment import assignment_cost, hungarian  # noqa: E402


def brute_force(cost):
    n, m = len(cost), len(cost[0])
    return min(
        sum(cost[i][p[i]] for i in range(n))
        for p in itertools.permutations(range(m), n)
    )


@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_on_random_rectangular_instances(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 5)
    m = rng.randint(n, 6)
    cost = [[rng.random() for _ in range(m)] for _ in range(n)]
    columns = hungarian(cost)
    assert len(columns) == n and len(set(columns)) == n
    assert assignment_cost(cost, columns) == pytest.approx(brute_force(cost))


def test_forbidden_pairs_and_shape_errors():
    assert hungarian([]) == []
    assert hungarian([[1.0, math.inf], [math.inf, 1.0]]) == [0, 1]
    with pytest.raises(ValueError, match="forbidden"):
        hungarian([[math.inf, math.inf], [1.0, 1.0]])
    with pytest.raises(ValueError, match="more rows"):
        hungarian([[1.0], [1.0]])
    with pytest.raises(ValueError, match="same length"):
        hungarian([[1.0, 2.0], [1.0]])


def test_formation_slots_use_the_shared_implementation():
    np = pytest.importorskip("numpy")
    from pymapf.swarm.formation import assign_slots

    positions = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    targets = np.array([[10.0, 1.0], [0.0, 1.0], [5.0, 1.0]])
    assert list(assign_slots(positions, targets)) == [1, 2, 0]
