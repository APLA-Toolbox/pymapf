# AGENTS.md

## Cursor Cloud specific instructions

PyMAPF is a self-contained Python multi-agent path finding library (no servers, DBs, or
external services). "Running the app" means executing a planner and rendering a matplotlib
animation. There are two planner families: centralized (`pymapf/centralized`, Space-Time
cooperative A*) and decentralized (`pymapf/decentralized`, NMPC and Velocity Obstacles).

### Environment / interpreter
- The pinned deps (`numpy==1.21.4`, `scipy==1.7.2`, `matplotlib==3.5.0`) only ship wheels
  for Python <= 3.10. System Python is 3.12, so the project runs in a **Python 3.10 uv venv
  at `.venv`** (created by the startup update script). Activate with `source .venv/bin/activate`
  (or prefix commands with `.venv/bin/`). Do not install into system Python 3.12 — the pins
  fail to build there.

### Headless gotchas (important)
- Always run with `MPLBACKEND=Agg` (no display in the VM).
- The centralized `CooperativeAStar.visualize()` (and `Animator.show()`) calls `plt.show()`,
  which **blocks indefinitely headless**. The A* planner itself is fast (~seconds). To render
  the centralized planner without hanging, save via the animator directly and skip `.show()`:
  `Animator(world, paths, agents, max(sim.searches_sim_times)).save("out")`.
- Because of the above, `tests/test_cooperative_astar_manager.py` hangs: `test_cooperative_astar_sim`
  passes, but `test_cooperative_astar_sim_diagonals` and `test_cooperative_astar_viz` call
  `visualize()` and block on `plt.show()`. Deselect them when running the suite headless, e.g.
  `pytest --deselect tests/test_cooperative_astar_manager.py::test_cooperative_astar_sim_diagonals --deselect tests/test_cooperative_astar_manager.py::test_cooperative_astar_viz`.
- The decentralized NMPC/Velocity-Obstacle `visualize()` do NOT call `plt.show()` and work
  headless (they use a `plt.pause` loop + `FuncAnimation.save`). `ffmpeg` is required for
  `.gif` output and is available in the VM.

### Timing
- NMPC simulations are heavy (scipy optimization per step): ~30-70s per `run_simulation`, so
  `tests/test_nmpc.py` takes ~3 min. Velocity-Obstacle tests take ~20s. Use generous timeouts.

### Commands
- Lint (matches CI): `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv`
- Tests: `MPLBACKEND=Agg pytest` (deselect the two blocking A* viz tests above).
- Run examples: `MPLBACKEND=Agg python scripts/switch_positions_nmpc.py` (NMPC, slow) and the
  library snippets in `README.md`. Note `main.py` is stale (imports a non-existent `src/` package).
