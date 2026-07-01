# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Centralized MAPF framework** (`pymapf.core`): an algorithm-agnostic layer
  that makes PyMAPF usable as a library long term.
  - `GridMap`: a deterministic, explicit occupancy grid (build a specific
    scenario instead of the random-wall-only `World`); includes
    `GridMap.from_world`.
  - `MAPFProblem`, `Agent`, `Solution` (with `makespan`, `sum_of_costs`,
    `is_valid`, `first_conflict`), and `Constraints`.
  - Pluggable heuristics (`manhattan`, `euclidean`, `chebyshev`, `octile`) with
    name/callable resolution, replacing the global `common.HEURISTIC` flag.
  - Abstract `MAPFSolver` plus a name-based solver **registry**
    (`register_solver`, `get_solver`, `available_solvers`) so new algorithms are
    discoverable and swappable.
  - Conflict detection utilities (`find_first_conflict`, `Conflict`).
- **New algorithms** (`pymapf.algorithms`):
  - `space_time_astar`: a constraint-aware, provably terminating low-level
    space-time A* shared by the solvers.
  - `PrioritizedPlanning` (`"prioritized"`): cooperative A* with space-time
    reservations.
  - `ConflictBasedSearch` (`"cbs"`): the canonical two-level optimal
    (sum-of-costs) MAPF algorithm.
- Top-level convenience API: `pymapf.solve(problem, algorithm="cbs", **kwargs)`
  and re-exports of the core framework types.
- Deterministic test suite for the new modules (heuristics, grid, low-level
  search, prioritized planning, CBS, and the solver registry).

### Changed

- `pymapf.__version__` bumped to `0.2.0`.

### Notes

- The existing reactive/decentralized planners (`MultiAgentNMPC`,
  `MultiAgentVelocityObstacle`) and the legacy `CooperativeAStar` are unchanged
  and remain available.
