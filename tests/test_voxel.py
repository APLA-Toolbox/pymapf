"""Three-dimensional grids: the solvers run unchanged, the geometry generalises."""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

import pymapf
from pymapf import Agent, MAPFProblem, VoxelGrid, available_solvers, build_scenario
from pymapf.core.heuristics import (
    chebyshev,
    euclidean,
    manhattan,
    octile,
    true_distance,
)
from pymapf.scenarios import SCENARIO_BUILDERS, to_ascii

FAST_KWARGS = {
    "lns": {"time_limit": 0.5},
    "lacam": {"time_limit": 5.0},
    "cbs": {"time_limit": 20.0},
    "wcbs": {"time_limit": 20.0},
}


def box(depth=3, height=3, width=3, blocked=()):
    voxels = [[[0] * width for _ in range(height)] for _ in range(depth)]
    for z, r, c in blocked:
        voxels[z][r][c] = 1
    return VoxelGrid(voxels)


# --------------------------------------------------------------------------
# the grid itself
# --------------------------------------------------------------------------


def test_voxel_grid_shape_and_free_count():
    grid = box(2, 3, 4, blocked=[(0, 0, 0), (1, 2, 3)])
    assert grid.shape == (2, 3, 4)
    assert (grid.depth, grid.height, grid.width) == (2, 3, 4)
    assert grid.dimension == 3
    assert grid.free_cells == 2 * 3 * 4 - 2
    assert not grid.is_free((0, 0, 0))
    assert grid.is_free((0, 0, 1))
    assert not grid.is_free((2, 0, 0))  # out of bounds is not free
    assert len(list(grid.cells())) == 24
    assert grid.layer(1)[2][3] == 1 and grid.layer(1)[0][0] == 0


@pytest.mark.parametrize(
    "voxels",
    [
        [],
        [[]],
        [[[]]],
        [[[0, 0]], [[0]]],  # layers of different widths
        [[[0], [0]], [[0]]],  # layers of different heights
    ],
)
def test_voxel_grid_rejects_ragged_or_empty_input(voxels):
    with pytest.raises(ValueError):
        VoxelGrid(voxels)


def test_six_face_neighbours_in_the_open_and_fewer_at_the_edge():
    grid = box()
    assert len(grid.neighbors((1, 1, 1))) == 6
    assert len(grid.neighbors((0, 0, 0))) == 3
    assert (1, 0, 0) in grid.neighbors((0, 0, 0))


def test_twenty_six_neighbours_with_diagonals():
    grid = box()
    diagonal = grid.neighbors((1, 1, 1), allow_diagonals=True)
    assert len(diagonal) == 26
    assert (0, 0, 0) in diagonal and (2, 2, 2) in diagonal


def test_diagonals_cannot_cut_a_blocked_face_edge_or_corner():
    """A move that changes several axes needs every axis-aligned part of it
    free -- the 3D reading of the planar corner-cutting rule."""
    centre = (1, 1, 1)
    # Blocking the +layer face refuses every diagonal that goes up.
    grid = box(blocked=[(2, 1, 1)])
    up = [n for n in grid.neighbors(centre, allow_diagonals=True) if n[0] == 2]
    assert up == []
    # An edge move (two axes) needs both its single-axis parts free.
    grid = box(blocked=[(1, 2, 1)])
    moves = grid.neighbors(centre, allow_diagonals=True)
    assert (1, 2, 2) not in moves  # would slide along the blocked row face
    assert (1, 1, 2) in moves  # the plain face move is unaffected
    assert (2, 1, 2) in moves  # an edge move not touching the blocked face
    # A corner move (three axes) needs all three faces and all three edges.
    grid = box(blocked=[(2, 2, 1)])  # one of the *edges* of the (2,2,2) corner
    assert (2, 2, 2) not in grid.neighbors(centre, allow_diagonals=True)


# --------------------------------------------------------------------------
# heuristics generalise
# --------------------------------------------------------------------------


def test_geometric_heuristics_take_any_dimension():
    assert manhattan((0, 0, 0), (1, 2, 3)) == 6
    assert chebyshev((0, 0, 0), (1, 2, 3)) == 3
    assert euclidean((0, 0, 0), (0, 3, 4)) == pytest.approx(5.0)
    # Planar results are unchanged.
    assert manhattan((0, 0), (2, 3)) == 5
    assert octile((0, 0), (3, 3)) == pytest.approx(3 * 2**0.5)
    assert octile((0, 0), (3, 0)) == 3
    # Octile on three axes: planar formula on the two largest deltas, plus the rest.
    assert octile((0, 0, 0), (3, 3, 1)) == pytest.approx(3 * 2**0.5 + 1)


def test_true_distance_runs_on_a_voxel_grid():
    grid = box(blocked=[(1, 1, 1)])
    h = true_distance(grid, (2, 2, 2))
    assert h((0, 0, 0), (2, 2, 2)) == 6
    assert h((2, 2, 2), (2, 2, 2)) == 0


# --------------------------------------------------------------------------
# every solver, unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", sorted(available_solvers()))
def test_every_solver_plans_in_three_dimensions(algorithm):
    scenario = build_scenario(
        "empty_volume", depth=3, height=5, width=5, n_agents=4, seed=1
    )
    solution = pymapf.solve(
        scenario.to_problem(), algorithm, **FAST_KWARGS.get(algorithm, {})
    )
    assert solution is not None, algorithm
    assert solution.is_valid()
    for agent in scenario.agents:
        assert solution.paths[agent.name][0] == agent.start
        assert solution.paths[agent.name][-1] == agent.goal
        for cell in solution.paths[agent.name]:
            assert len(cell) == 3 and scenario.grid.is_free(cell)


def test_cbs_is_optimal_in_three_dimensions():
    grid = box(3, 3, 3)
    problem = MAPFProblem(grid, [Agent("a", (0, 0, 0), (2, 2, 2))])
    solution = pymapf.solve(problem, "cbs")
    assert solution.sum_of_costs == 6  # the Manhattan distance, no wall in the way


def test_the_third_axis_is_a_way_around():
    """Two agents swapping along a 1-wide corridor deadlock on a plane; with a
    layer above they pass."""
    voxels = [[[0, 0, 0]], [[0, 0, 0]]]  # 2 layers x 1 row x 3 cols
    grid = VoxelGrid(voxels)
    problem = MAPFProblem(
        grid, [Agent("a", (0, 0, 0), (0, 0, 2)), Agent("b", (0, 0, 2), (0, 0, 0))]
    )
    solution = pymapf.solve(problem, "cbs")
    assert solution is not None and solution.is_valid()
    used_layers = {cell[0] for path in solution.paths.values() for cell in path}
    assert used_layers == {0, 1}


def test_problem_validation_covers_voxel_cells():
    grid = box(blocked=[(1, 1, 1)])
    with pytest.raises(ValueError):
        MAPFProblem(grid, [Agent("a", (1, 1, 1), (0, 0, 0))])
    with pytest.raises(ValueError):
        MAPFProblem(grid, [Agent("a", (0, 0, 0), (3, 0, 0))])


# --------------------------------------------------------------------------
# scenario families
# --------------------------------------------------------------------------


def test_three_dimensional_families_are_registered():
    assert {"empty_volume", "random_blocks", "stacked_floors"} <= set(SCENARIO_BUILDERS)


def test_stacked_floors_forces_every_agent_through_a_shaft():
    scenario = build_scenario("stacked_floors", floors=3, n_agents=3, shafts=1, seed=2)
    for agent in scenario.agents:
        assert agent.start[0] != agent.goal[0], "start and goal share a floor"
    solution = pymapf.solve(scenario.to_problem(), "pibt")
    assert solution is not None and solution.is_valid()
    # Slabs sit at the odd layers; each is pierced by exactly one shaft, and
    # every path changes floor, so every path crosses at least one of them.
    grid = scenario.grid
    slabs = range(1, grid.depth, 2)
    shaft_cells = {
        (z, r, c)
        for z in slabs
        for r in range(grid.height)
        for c in range(grid.width)
        if grid.is_free((z, r, c))
    }
    assert len(shaft_cells) == len(slabs)
    for path in solution.paths.values():
        assert any(cell in shaft_cells for cell in path)


def test_random_blocks_validates_density_and_is_seeded():
    with pytest.raises(ValueError):
        build_scenario("random_blocks", density=1.0)
    a = build_scenario("random_blocks", seed=4)
    b = build_scenario("random_blocks", seed=4)
    assert to_ascii(a) == to_ascii(b)
    assert to_ascii(a) != to_ascii(build_scenario("random_blocks", seed=5))


def test_ascii_rendering_of_a_volume_is_one_block_per_layer():
    scenario = build_scenario(
        "empty_volume", depth=2, height=2, width=3, n_agents=1, seed=0
    )
    text = to_ascii(scenario)
    blocks = [block for block in text.split("\n\n") if block.strip()]
    assert len(blocks) == 2
    assert blocks[0].startswith("layer 0")
    assert blocks[1].startswith("layer 1")


def test_scenario_repr_reports_three_dimensions():
    scenario = build_scenario("empty_volume", depth=2, height=3, width=4, n_agents=1)
    assert "2x3x4" in repr(scenario)


# --------------------------------------------------------------------------
# what the rest of the library does with a volume
# --------------------------------------------------------------------------


def test_trajectories_follow_a_three_dimensional_plan():
    # pymapf.kinodynamic lands in a sibling PR; until both are on main this
    # integration check is skipped rather than failing on an import.
    try:
        from pymapf.kinodynamic import KinematicLimits, plan_trajectories
    except ImportError:
        pytest.skip("pymapf.kinodynamic not available on this branch")

    scenario = build_scenario(
        "empty_volume", depth=3, height=4, width=4, n_agents=3, seed=3
    )
    solution = pymapf.solve(scenario.to_problem(), "cbs", time_limit=20.0)
    trajectories = plan_trajectories(
        solution, limits=KinematicLimits(1.0, safety_distance=0.5)
    )
    assert trajectories.min_separation(0.05)[0] >= 0.5 - 0.05
    assert len(trajectories["A"].position_at(0.0)) == 3


def test_planar_plots_refuse_a_volume_and_point_at_the_3d_one():
    pytest.importorskip("matplotlib")
    from pymapf import viz

    scenario = build_scenario("empty_volume", n_agents=2, seed=0)
    solution = pymapf.solve(scenario.to_problem(), "pibt")
    with pytest.raises(TypeError, match="plot_solution_3d"):
        viz.plot_solution(solution, scenario)
    ax = viz.plot_solution_3d(solution, scenario)
    assert ax.name == "3d"
    with pytest.raises(TypeError):
        viz.plot_solution_3d(solution, build_scenario("empty_room", n_agents=2))


def test_rl_environment_says_no_to_volumes_clearly():
    pytest.importorskip("numpy")
    from pymapf.rl import MAPFEnv

    scenario = build_scenario("empty_volume", n_agents=2, seed=0)
    with pytest.raises(NotImplementedError, match="planar"):
        MAPFEnv(scenario)
