"""Decentralized navigation: ORCA, buffered Voronoi cells, potential fields, social forces.

The constraint methods are held to their guarantee -- never closer than the
separation distance -- on every run, in two and three dimensions. Arrival is
asserted where it was measured to hold and recorded where it was measured to
fail: the symmetric deadlocks are real behaviour of these laws, not of this
implementation, and a test that says so beats a docstring that does.
"""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

np = pytest.importorskip("numpy")

from pymapf.swarm import (  # noqa: E402
    ORCA,
    BufferedVoronoi,
    NavigationBehavior,
    PotentialField,
    SocialForce,
    SwarmParams,
    SwarmSimulator,
    SwarmState,
    available_behaviors,
    circle_swap,
    get_behavior,
    project_onto_polytope,
)

NAVIGATION = ["orca", "buffered_voronoi", "potential_field", "social_force"]
SEPARATION = 1.0


def params(**changes):
    base = dict(
        separation_distance=SEPARATION,
        sensing_range=15.0,
        cruise_speed=1.5,
        max_speed=2.0,
        max_acceleration=4.0,
    )
    base.update(changes)
    return SwarmParams(**base)


def run(behavior, positions, goals, steps=400, prm=None, **kwargs):
    positions = np.asarray(positions, dtype=float)
    goals = np.asarray(goals, dtype=float)
    state = SwarmState(positions, np.zeros_like(positions))
    simulator = SwarmSimulator(
        behavior, initial=state, params=prm or params(), dt=0.1, goals=goals, **kwargs
    )
    result = simulator.run(steps=steps)
    remaining = np.linalg.norm(result.final.positions - goals, axis=1)
    started = np.linalg.norm(positions - goals, axis=1)
    return {
        "result": result,
        "arrived": int((remaining < 0.5).sum()),
        "progress": 1.0 - remaining.sum() / started.sum(),
        "min_separation": min(result.metrics.min_distance, default=float("inf")),
    }


def swap(behavior, n, dimension=2, steps=400, **kwargs):
    state, goals = circle_swap(n, radius=10.0, dimension=dimension)
    return run(behavior, state.positions, goals, steps=steps, **kwargs)


# --------------------------------------------------------------------------
# registry and base class
# --------------------------------------------------------------------------


def test_navigation_laws_are_registered_like_every_other_behavior():
    assert set(NAVIGATION) <= set(available_behaviors())
    for name in NAVIGATION:
        assert isinstance(get_behavior(name), NavigationBehavior)
    assert get_behavior("social_force").output == "acceleration"
    for name in ("orca", "buffered_voronoi", "potential_field"):
        assert get_behavior(name).output == "velocity"


def test_goals_default_to_the_migration_point_and_must_match_the_swarm():
    state = SwarmState(np.zeros((3, 2)), np.zeros((3, 2)))
    behavior = ORCA(params=params(migration_point=(4.0, 1.0)))
    behavior.reset(state)
    assert np.allclose(behavior.goals, [[4.0, 1.0]] * 3)
    with pytest.raises(ValueError, match="goals"):
        ORCA(goals=[[1.0, 1.0]]).reset(state)


def test_preferred_velocity_points_at_the_goal_and_settles():
    behavior = PotentialField(goals=[[10.0, 0.0]], tie_break=0.0, params=params())
    far = SwarmState(np.array([[0.0, 0.0]]), np.zeros((1, 2)))
    behavior.reset(far)
    v = behavior.preferred_velocity(far, 0)
    assert np.allclose(v, [1.5, 0.0])  # cruise speed, straight at it
    near = SwarmState(np.array([[9.5, 0.0]]), np.zeros((1, 2)))
    assert np.linalg.norm(behavior.preferred_velocity(near, 0)) < 1.5  # ramping down
    there = SwarmState(np.array([[9.9, 0.0]]), np.zeros((1, 2)))
    assert np.allclose(behavior.preferred_velocity(there, 0), 0.0)
    assert behavior.arrived(there).tolist() == [True]


def test_circle_swap_is_antipodal_in_two_and_three_dimensions():
    for dimension in (2, 3):
        state, goals = circle_swap(10, radius=7.0, dimension=dimension, jitter=0.0)
        assert state.positions.shape == (10, dimension)
        assert np.allclose(goals, -state.positions)
        assert np.allclose(np.linalg.norm(state.positions, axis=1), 7.0)
    with pytest.raises(ValueError):
        circle_swap(4, dimension=4)


# --------------------------------------------------------------------------
# the exact projection every constraint method relies on
# --------------------------------------------------------------------------


def test_projection_onto_a_wedge_and_a_ball():
    normals = np.array([[1.0, 0.0], [0.0, 1.0]])  # first quadrant
    offsets = np.zeros(2)
    for target, expected in [
        ((-3, -4), (0, 0)),
        ((-3, 5), (0, 5)),
        ((2, -1), (2, 0)),
        ((1, 1), (1, 1)),
    ]:
        x, ok = project_onto_polytope(np.array(target, float), normals, offsets)
        assert ok and np.allclose(x, expected)
    x, ok = project_onto_polytope(np.array([5.0, 5.0]), normals, offsets, radius=1.0)
    assert ok and np.allclose(x, [1 / 2**0.5] * 2)


def test_projection_reports_an_empty_intersection():
    normals = np.array([[1.0, 0.0], [-1.0, 0.0]])
    offsets = np.array([1.0, 1.0])  # x >= 1 and x <= -1
    _, ok = project_onto_polytope(np.zeros(2), normals, offsets)
    assert not ok
    _, ok = project_onto_polytope(
        np.zeros(2), np.array([[1.0, 0.0]]), np.array([5.0]), radius=1.0
    )
    assert not ok  # the plane misses the ball


def test_projection_matches_a_general_purpose_solver_in_two_to_four_dimensions():
    minimize = pytest.importorskip("scipy.optimize").minimize
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(120):
        d = int(rng.choice([2, 3, 4]))
        m = int(rng.integers(1, 10))
        target = rng.normal(0, 3, d)
        interior = rng.normal(0, 1, d)
        normals = rng.normal(0, 1, (m, d))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        offsets = normals @ interior - rng.uniform(0.0, 1.5, m)
        # A ball that contains the interior point, so the intersection is non-empty.
        radius = (
            None
            if rng.random() < 0.5
            else float(np.linalg.norm(interior) + rng.uniform(0.5, 3.0))
        )
        x, ok = project_onto_polytope(target, normals, offsets, radius)
        assert ok
        assert np.all(normals @ x - offsets >= -1e-7)
        if radius is not None:
            assert np.linalg.norm(x) <= radius + 1e-7
        constraints = [{"type": "ineq", "fun": lambda v: normals @ v - offsets}]
        if radius is not None:
            constraints.append({"type": "ineq", "fun": lambda v: radius**2 - v @ v})
        start = (
            interior
            if radius is None
            else interior * min(1.0, 0.9 * radius / np.linalg.norm(interior))
        )
        reference = minimize(
            lambda v: np.sum((v - target) ** 2),
            start,
            constraints=constraints,
            method="SLSQP",
        )
        if reference.success:
            assert (
                np.linalg.norm(x - target)
                <= np.linalg.norm(reference.x - target) + 1e-6
            )
            checked += 1
    assert checked > 80


# --------------------------------------------------------------------------
# ORCA
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", [2, 3])
def test_orca_pair_passes_and_keeps_its_distance(dimension):
    a = np.zeros(dimension)
    b = np.zeros(dimension)
    a[0], b[0] = -6.0, 6.0
    b[-1] += 0.3  # a slight offset, so the encounter is not perfectly symmetric
    out = run("orca", [a, b], [b, a], steps=200)
    assert out["arrived"] == 2
    assert out["min_separation"] >= SEPARATION - 1e-6


@pytest.mark.parametrize("n, steps", [(4, 600), (8, 600), (24, 800)])
def test_orca_negotiates_the_circle_swap(n, steps):
    out = swap("orca", n, steps=steps)
    assert out["result"].metrics.collisions == 0
    assert out["min_separation"] >= SEPARATION - 1e-6
    assert out["arrived"] == n


def test_orca_in_three_dimensions_on_a_sphere():
    out = swap("orca", 16, dimension=3, steps=600)
    assert out["min_separation"] >= SEPARATION - 1e-6
    assert out["arrived"] == 16


def test_orca_twelve_agent_ring_deadlocks_but_never_collides():
    """On record: twelve agents converging on one point form a ring that
    ORCA does not break, with or without tie-breaking noise. Eight and
    twenty-four flow; this count does not. It is the symmetric deadlock the
    literature describes, and it costs no collision."""
    out = swap("orca", 12, steps=600)
    assert out["min_separation"] >= SEPARATION - 1e-6
    assert out["progress"] < 0.6


def test_orca_goes_around_an_obstacle_at_exactly_its_radius():
    prm = params(obstacles=[((0.0, 0.2), 1.0)])
    out = run("orca", [[-6.0, 0.0]], [[6.0, 0.0]], steps=300, prm=prm)
    assert out["arrived"] == 1
    track = np.array([s.positions[0] for s in out["result"].history])
    clearance = np.linalg.norm(track - [0.0, 0.2], axis=1).min() - 1.0
    assert clearance >= SEPARATION / 2 - 1e-6


# --------------------------------------------------------------------------
# buffered Voronoi cells
# --------------------------------------------------------------------------


def test_bvc_cell_always_contains_the_agent_and_excludes_its_neighbours():
    state, goals = circle_swap(9, radius=4.0)
    behavior = BufferedVoronoi(goals=goals, params=params())
    behavior.reset(state)
    for i in range(state.n):
        normals, offsets = behavior.cell(state, i)
        assert np.all(normals @ state.positions[i] - offsets >= -1e-9)
        for j in range(state.n):
            if j != i:
                assert np.any(normals @ state.positions[j] - offsets < 0)


@pytest.mark.parametrize("n, dimension", [(4, 2), (12, 2), (16, 3)])
def test_bvc_never_collides(n, dimension):
    out = swap("buffered_voronoi", n, dimension=dimension, steps=500)
    assert out["result"].metrics.collisions == 0
    assert out["min_separation"] >= SEPARATION - 1e-6


def test_bvc_passes_an_offset_pair_and_staggered_lanes():
    out = run("buffered_voronoi", [[-6, 0], [6, 0.3]], [[6, 0], [-6, 0.3]], steps=400)
    assert out["arrived"] == 2 and out["min_separation"] >= SEPARATION
    starts = [[-6, 0], [6, 1.5], [-6, 3], [6, 4.5]]
    goals = [[6, 0], [-6, 1.5], [6, 3], [-6, 4.5]]
    out = run("buffered_voronoi", starts, goals, steps=500)
    assert out["arrived"] == 4 and out["min_separation"] >= SEPARATION


def test_bvc_deadlocks_on_a_symmetric_crossing_as_the_paper_says():
    """On record: agents whose straight lines cross at one point wedge
    against their cell boundaries. The detour helps a little and does not
    resolve it; neither run collides."""
    with_detour = swap("buffered_voronoi", 12, steps=800)
    without = swap("buffered_voronoi", 12, steps=800, detour=False)
    for out in (with_detour, without):
        assert out["min_separation"] >= SEPARATION - 1e-6
        assert out["progress"] < 0.6
    assert with_detour["progress"] >= without["progress"]


def test_bvc_goes_around_an_obstacle_with_room_to_spare():
    prm = params(obstacles=[((0.0, 0.2), 1.0)])
    out = run("buffered_voronoi", [[-6.0, 0.0]], [[6.0, 0.0]], steps=300, prm=prm)
    assert out["arrived"] == 1
    track = np.array([s.positions[0] for s in out["result"].history])
    assert np.linalg.norm(track - [0.0, 0.2], axis=1).min() - 1.0 >= SEPARATION / 2


# --------------------------------------------------------------------------
# force methods
# --------------------------------------------------------------------------


@pytest.mark.parametrize("behavior", ["potential_field", "social_force"])
@pytest.mark.parametrize("n", [8, 12])
def test_force_methods_solve_the_circle_swap(behavior, n):
    out = swap(behavior, n, steps=600)
    assert out["arrived"] == n
    assert out["result"].metrics.collisions == 0


def test_potential_field_stops_in_the_local_minimum_behind_an_obstacle():
    """Khatib's known failure: the goal directly behind an obstacle puts a
    minimum of the summed field in front of it, and the agent parks there."""
    prm = params(obstacles=[((0.0, 0.0), 1.0)])
    out = run(
        "potential_field",
        [[-6.0, 0.0]],
        [[6.0, 0.0]],
        steps=400,
        prm=prm,
        tie_break=0.0,
    )
    assert out["arrived"] == 0
    final = out["result"].final.positions[0]
    assert final[0] < 0  # stuck on the near side


def test_social_force_is_second_order():
    behavior = SocialForce(goals=[[10.0, 0.0]], params=params(), tie_break=0.0)
    state = SwarmState(np.array([[0.0, 0.0]]), np.zeros((1, 2)))
    behavior.reset(state)
    command = behavior.command(state, 0)
    # From rest, the driving force is (preferred - 0) / tau, capped.
    assert np.allclose(command, [min(1.5 / behavior.tau, 4.0), 0.0])


@pytest.mark.parametrize("behavior", NAVIGATION)
def test_every_law_runs_in_three_dimensions(behavior):
    out = swap(behavior, 6, dimension=3, steps=200)
    assert out["result"].final.positions.shape == (6, 3)
    assert np.isfinite(out["result"].final.positions).all()
