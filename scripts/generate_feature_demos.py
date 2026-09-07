"""Render the demos for the quadrotor docking, 3D, navigation and lifelong layers.

One animation and one figure per feature, written to ``.docs/assets`` in the
same theme as the rest of the gallery, so the README and the site show what
each layer does rather than describe it::

    python scripts/generate_feature_demos.py             # everything
    python scripts/generate_feature_demos.py --only quadrotor  # one of them

Every figure is produced from a seeded run, so regenerating gives the same
picture and the numbers in the captions match the ones in the README.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import pymapf  # noqa: E402
from pymapf import viz  # noqa: E402
from pymapf.viz import theme as theme_module  # noqa: E402

THEME = "dark"
ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".docs", "assets"
)


def _step(label: str) -> float:
    print("  %-46s" % label, end="", flush=True)
    return time.perf_counter()


def _done(started: float, path: str) -> None:
    size = os.path.getsize(path) / 1024.0
    print(
        "%6.1fs  %s (%.0f kB)"
        % (time.perf_counter() - started, os.path.basename(path), size)
    )


def _save_gif(animation, path: str, fps: int, dpi: int = 90) -> str:
    import matplotlib.pyplot as plt

    viz.save_animation(animation, path, fps=fps, dpi=dpi)
    plt.close("all")
    return path


def _solve(problem, preferred: str = "cbs", time_limit: float = 60.0):
    solution = pymapf.solve(problem, preferred, time_limit=time_limit)
    if solution is None:
        solution = pymapf.solve(problem, "lacam", time_limit=time_limit)
    return solution


# --------------------------------------------------------------------------
# 1. quadrotor docking: assignment, routes and kinematic trajectories in 3D
# --------------------------------------------------------------------------


def _docking_instance():
    """A fleet spread over a field of pads with a building in the way.

    Of a few seeded fleets, the one whose optimal routes are longest is the
    one worth watching; every candidate is solved by CBS, so the routes are
    optimal for the assignment whichever seed wins.
    """
    from pymapf.aerial import Airspace, hovering_fleet, plan_docking
    from pymapf.kinodynamic import KinematicLimits

    airspace = Airspace.build(
        cells=(16, 12, 7),
        cell_size=1.5,
        pads=[(3, c) for c in (3, 5, 7, 9, 11)] + [(8, c) for c in (3, 5, 7, 9, 11)],
        obstacles=[((5, 6, 0), (6, 9, 3)), ((1, 13, 0), (1, 13, 5))],
        floor=2,
    )
    best = None
    for seed in range(6):
        fleet = hovering_fleet(airspace, n=8, seed=seed, min_layer=3, radius=0.35)
        try:
            plan = plan_docking(
                fleet,
                airspace,
                limits=KinematicLimits(v_max=2.0, a_max=1.5),
                clearance=0.3,
                algorithm="cbs",
                time_limit=20.0,
            )
        except RuntimeError:
            continue
        if (
            best is None
            or plan.summary()["route_cost"] > best[2].summary()["route_cost"]
        ):
            best = (airspace, fleet, plan)
    return best


def _obstacle_voxels(airspace):
    """Obstacles as an ``(x, y, z)`` boolean array: what is blocked at or
    above the flight floor, and below it only the columns of those (the
    floor itself is a rule, not a thing to draw)."""
    depth, height, width = airspace.grid.shape
    filled = np.zeros((width, height, depth), dtype=bool)
    for r in range(height):
        for c in range(width):
            for z in range(airspace.floor, depth):
                if not airspace.grid.is_free((z, r, c)):
                    filled[c, r, z] = True
            if filled[c, r, airspace.floor]:
                filled[c, r, 1 : airspace.floor] = True
    return filled


def _draw_world(ax, airspace, plan, resolved, colors):
    cs = airspace.cell_size
    x_len, y_len, z_len = airspace.extent
    ax.set_facecolor(resolved.plane)
    ax.figure.patch.set_facecolor(resolved.plane)
    # Draw in the order given, not by depth: the ground stays under everything.
    ax.computed_zorder = False
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0, 0, 0, 0))
        pane.pane.set_edgecolor(resolved.grid)
        pane._axinfo["grid"]["color"] = resolved.grid
    ax.set_xlim(-cs, x_len + cs)
    ax.set_ylim(-cs, y_len + cs)
    ax.set_zlim(0, z_len + cs)
    ax.set_box_aspect((x_len + 2 * cs, y_len + 2 * cs, z_len + cs), zoom=1.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # the ground, as a surface so that what is drawn above it stays visible
    gx, gy = np.meshgrid([-cs / 2, x_len + cs / 2], [-cs / 2, y_len + cs / 2])
    ax.plot_surface(
        gx,
        gy,
        np.zeros_like(gx),
        color=resolved.surface,
        alpha=0.9,
        shade=False,
        zorder=0,
    )
    # the pads, coloured by whoever lands there
    landing = {station: name for name, station in plan.assignment.items()}
    for station in plan.stations:
        x, y, _ = station.position
        color = colors.get(landing.get(station.name), resolved.axis)
        h = 0.42 * cs
        ax.plot(
            [x - h, x + h, x + h, x - h, x - h],
            [y - h, y - h, y + h, y + h, y - h],
            [0.12 * cs] * 5,
            color=color,
            linewidth=2.2,
            zorder=3,
        )
    # obstacles: one translucent cube per blocked voxel, drawn explicitly so
    # that the draw order above holds for them too
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    filled = _obstacle_voxels(airspace)
    faces = []
    for c, r, z in zip(*np.nonzero(filled)):
        x0, y0, z0 = (c - 0.5) * cs, (r - 0.5) * cs, (z - 0.5) * cs
        x1, y1, z1 = x0 + cs, y0 + cs, z0 + cs
        corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
        for quad in (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
            (0, 3, 7, 4),
        ):
            faces.append([corners[i] for i in quad])
    if faces:
        # Light enough to survive GIF palette quantisation next to the trails.
        blocks = Poly3DCollection(
            faces,
            facecolors=resolved.axis,
            edgecolors=resolved.muted,
            linewidths=0.6,
            alpha=0.75,
        )
        blocks.set_zorder(1)
        ax.add_collection3d(blocks)


def quadrotor(out) -> None:
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(THEME)
    started = _step("quadrotor: assign, route, schedule")
    airspace, fleet, plan = _docking_instance()
    summary = plan.summary()
    names = [q.name for q in fleet]
    colors = resolved.color_map(names)
    print(
        "%.1fs  makespan %.1fs, min separation %.2f (need %.2f)"
        % (
            time.perf_counter() - started,
            plan.makespan,
            summary["min_separation"],
            summary["required_separation"],
        )
    )

    # -- the animation --------------------------------------------------------
    started = _step("quadrotor: animating the docking")
    dt = 0.1
    makespan = plan.makespan
    frames = int(math.ceil(makespan / dt)) + 20
    figure = plt.figure(figsize=(7.4, 5.6))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=0.9)
    ax = figure.add_subplot(111, projection="3d")
    ax.view_init(elev=36, azim=-56)
    _draw_world(ax, airspace, plan, resolved, colors)
    for q in fleet:
        x, y, z = q.position
        ax.plot([x, x], [y, y], [0, z], color=colors[q.name], linewidth=0.5, alpha=0.25)
    bodies = {
        name: ax.plot(
            [],
            [],
            [],
            marker="o",
            markersize=10,
            color=colors[name],
            linestyle="none",
            zorder=6,
        )[0]
        for name in names
    }
    rotors = {
        name: ax.plot(
            [],
            [],
            [],
            marker="+",
            markersize=18,
            markeredgewidth=1.4,
            color=colors[name],
            linestyle="none",
            zorder=7,
        )[0]
        for name in names
    }
    trails = {
        name: ax.plot(
            [], [], [], color=colors[name], linewidth=2.6, alpha=0.95, zorder=4
        )[0]
        for name in names
    }
    readout = ax.text2D(
        0.02, 0.96, "", transform=ax.transAxes, fontsize=9, color=resolved.ink, va="top"
    )
    figure.suptitle(
        "plan_docking() · %d quadrotors, %d pads · Hungarian + CBS + MAPF-POST runs\n"
        "v_max %.1f m/s, a_max %.1f m/s², bodies %.2f m apart at least — measured %.2f"
        % (
            len(fleet),
            len(plan.stations),
            plan.limits.v_max,
            plan.limits.a_max,
            summary["required_separation"],
            summary["min_separation"],
        ),
        fontsize=9.5,
        color=resolved.ink,
    )
    trail_seconds = 3.0
    closest_so_far = [math.inf]
    sample_times, sample_positions = plan.trajectories.sample(dt)

    def update(frame):
        t = min(frame * dt, makespan)
        k = min(frame, len(sample_times) - 1)
        docked = 0
        for name in names:
            trajectory = plan.trajectories[name]
            x, y, z = trajectory.position_at(t)
            bodies[name].set_data_3d([x], [y], [z])
            rotors[name].set_data_3d([x], [y], [z])
            start = max(0, k - int(trail_seconds / dt))
            points = np.array(sample_positions[name][start : k + 1])
            trails[name].set_data_3d(points[:, 0], points[:, 1], points[:, 2])
            if t >= trajectory.arrival_time - 1e-9:
                docked += 1
        positions = np.array([plan.trajectories[name].position_at(t) for name in names])
        if len(names) > 1:
            diff = positions[:, None, :] - positions[None, :, :]
            dist = np.sqrt((diff**2).sum(axis=2)) + np.eye(len(names)) * 1e9
            closest_so_far[0] = min(closest_so_far[0], float(dist.min()))
        readout.set_text(
            "t = %5.1f s   docked %d/%d   closest so far %.2f m"
            % (t, docked, len(names), closest_so_far[0])
        )
        return (
            list(bodies.values())
            + list(rotors.values())
            + list(trails.values())
            + [readout]
        )

    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 * dt, blit=False
    )
    path = _save_gif(
        animation, out("animated-quadrotor-docking.gif"), fps=int(1 / dt), dpi=80
    )
    _done(started, path)

    _docking_figure(out, airspace, fleet, plan, summary, colors, resolved)


def _docking_figure(out, airspace, fleet, plan, summary, colors, resolved) -> None:
    import matplotlib.pyplot as plt

    # -- the figure: top view, altitude, separation -----------------------------
    started = _step("quadrotor: top view, altitude and separation")
    theme_module.apply(THEME)
    figure = plt.figure(figsize=(10.5, 4.8))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.45, wspace=0.2)
    top = figure.add_subplot(grid[:, 0])
    altitude = figure.add_subplot(grid[0, 1])
    separation = figure.add_subplot(grid[1, 1], sharex=altitude)

    cs = airspace.cell_size
    x_len, y_len, _ = airspace.extent
    top.set_facecolor(resolved.surface)
    top.set_xlim(-cs, x_len + cs)
    top.set_ylim(-cs, y_len + cs)
    top.set_aspect("equal")
    top.set_xticks([])
    top.set_yticks([])
    depth, height, width = airspace.grid.shape
    from matplotlib.patches import Rectangle

    for r in range(height):
        for c in range(width):
            if any(not airspace.grid.is_free((z, r, c)) for z in range(1, depth)):
                top.add_patch(
                    Rectangle(
                        ((c - 0.5) * cs, (r - 0.5) * cs),
                        cs,
                        cs,
                        facecolor=resolved.obstacle,
                        edgecolor="none",
                    )
                )
    landing = {station: name for name, station in plan.assignment.items()}
    for station in plan.stations:
        x, y, _ = station.position
        color = colors.get(landing.get(station.name), resolved.axis)
        top.add_patch(
            Rectangle(
                (x - 0.4 * cs, y - 0.4 * cs),
                0.8 * cs,
                0.8 * cs,
                facecolor="none",
                edgecolor=color,
                linewidth=1.6,
            )
        )
    times, positions = plan.trajectories.sample(0.05)
    for q in fleet:
        points = np.array(positions[q.name])
        top.plot(
            points[:, 0], points[:, 1], color=colors[q.name], linewidth=1.6, alpha=0.9
        )
        top.plot(
            q.position[0], q.position[1], marker="o", color=colors[q.name], markersize=7
        )
        top.annotate(
            q.name,
            (q.position[0], q.position[1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=colors[q.name],
        )
        pad = plan.station_of(q.name).position
        top.plot(
            [q.position[0], pad[0]],
            [q.position[1], pad[1]],
            color=colors[q.name],
            linewidth=0.7,
            linestyle=":",
            alpha=0.6,
        )
    top.set_title(
        "Top view: optimal assignment (dotted) and the routes flown", fontsize=10
    )

    for q in fleet:
        points = np.array(positions[q.name])
        altitude.plot(times, points[:, 2], color=colors[q.name], linewidth=1.3)
    altitude.set_ylabel("altitude (m)")
    altitude.set_title(
        "Altitude: hover, transit, vertical descent onto the pad", fontsize=10
    )
    altitude.tick_params(labelbottom=False)

    names_list = list(positions)
    stacked = np.array([positions[name] for name in names_list])  # (n, T, 3)
    diff = stacked[:, None, :, :] - stacked[None, :, :, :]
    dist = np.sqrt((diff**2).sum(axis=3))
    n = len(names_list)
    pairwise_min = np.array(
        [dist[:, :, k][~np.eye(n, dtype=bool)].min() for k in range(len(times))]
    )
    separation.plot(times, pairwise_min, color=resolved.agent_color(0), linewidth=1.4)
    separation.axhline(
        summary["required_separation"],
        color=resolved.muted,
        linewidth=0.9,
        linestyle="--",
    )
    separation.text(
        times[0],
        summary["required_separation"],
        " bodies touch",
        color=resolved.muted,
        fontsize=8,
        va="center",
    )
    separation.set_ylim(0, max(pairwise_min.max() * 1.05, 1.0))
    separation.set_ylabel("closest pair (m)")
    separation.set_xlabel("time (s)")
    separation.set_title(
        "Closest pair over the whole flight: never below %.2f m" % pairwise_min.min(),
        fontsize=10,
    )
    path = out("quadrotor-docking.png")
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    _done(started, path)


# --------------------------------------------------------------------------
# 2. three dimensions: routes threading a stack of floors
# --------------------------------------------------------------------------


def volume(out) -> None:
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    scenario = pymapf.build_scenario(
        "stacked_floors", floors=3, height=7, width=7, n_agents=6, shafts=2, seed=0
    )
    solution = _solve(scenario.to_problem())

    started = _step("3D: stacked floors, static")
    ax = viz.plot_solution_3d(
        solution, scenario, theme=THEME, title="stacked_floors · 6 agents, CBS"
    )
    path = out("solution-3d.png")
    ax.figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
    _done(started, path)

    started = _step("3D: stacked floors, orbit")
    ax = viz.plot_solution_3d(
        solution, scenario, theme=THEME, title="plot_solution_3d() · a volume is a map"
    )
    figure = ax.figure

    def update(frame):
        ax.view_init(elev=26.0, azim=-55.0 + 4.0 * frame)
        return []

    animation = FuncAnimation(figure, update, frames=90, interval=50, blit=False)
    path = _save_gif(animation, out("animated-3d.gif"), fps=20, dpi=80)
    _done(started, path)


# --------------------------------------------------------------------------
# 3. decentralized navigation: eight agents crossing a circle
# --------------------------------------------------------------------------


def _swap_run(law: str, n: int = 8, steps: int = 600):
    from pymapf.swarm import SwarmParams, SwarmSimulator, circle_swap

    state, goals = circle_swap(n=n, radius=10.0)
    params = SwarmParams(separation_distance=1.0, cruise_speed=1.5)
    result = SwarmSimulator(law, initial=state, params=params, goals=goals).run(
        steps=steps
    )
    return result, goals


def _arrived(result, goals, tolerance: float = 0.5) -> int:
    final = result.final.positions
    return int(np.sum(np.linalg.norm(final - goals, axis=1) <= tolerance))


def navigation(out) -> None:
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(THEME)

    # -- the animation: ORCA ---------------------------------------------------
    started = _step("navigation: ORCA circle swap")
    result, goals = _swap_run("orca")
    positions = result.positions()
    n = positions.shape[1]
    colors = [resolved.agent_color(i) for i in range(n)]
    min_distance = (
        min(result.metrics.min_distance) if result.metrics.min_distance else math.nan
    )

    figure, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set_facecolor(resolved.surface)
    ax.set_xlim(-11.5, 11.5)
    ax.set_ylim(-11.5, 11.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(n):
        ax.plot(
            goals[i, 0],
            goals[i, 1],
            marker="x",
            markersize=8,
            color=colors[i],
            alpha=0.8,
            markeredgewidth=1.6,
            zorder=2,
        )
    bodies = []
    for i in range(n):
        body = Circle((0, 0), 0.5, facecolor=colors[i], edgecolor="none", zorder=5)
        ax.add_patch(body)
        bodies.append(body)
    trails = [
        ax.plot([], [], color=colors[i], linewidth=1.6, alpha=0.5, zorder=4)[0]
        for i in range(n)
    ]
    clock = ax.text(
        0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left", fontsize=9
    )
    figure.suptitle(
        'SwarmSimulator("orca") · 8 agents swap sides\nmin separation %.2f · collisions %d'
        % (min_distance, result.metrics.collisions),
        fontsize=9.5,
        color=resolved.ink,
    )
    stride = 3
    frames = positions.shape[0] // stride

    def update(frame):
        k = min(frame * stride, positions.shape[0] - 1)
        for i in range(n):
            bodies[i].center = tuple(positions[k, i])
            start = max(0, k - 100)
            trails[i].set_data(
                positions[start : k + 1, i, 0], positions[start : k + 1, i, 1]
            )
        clock.set_text("t = %5.1f s" % (k * 0.1))
        return bodies + trails + [clock]

    animation = FuncAnimation(figure, update, frames=frames, interval=100, blit=True)
    path = _save_gif(animation, out("animated-orca.gif"), fps=20, dpi=90)
    _done(started, path)

    # -- the figure: the four laws on the same crossing -----------------------
    started = _step("navigation: four laws, one crossing")
    theme_module.apply(THEME)
    laws = [
        ("orca", "ORCA"),
        ("buffered_voronoi", "Buffered Voronoi cells"),
        ("potential_field", "Potential field"),
        ("social_force", "Social forces"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 8.4))
    for ax, (law, label) in zip(axes.ravel(), laws):
        result, goals = _swap_run(law)
        positions = result.positions()
        n = positions.shape[1]
        ax.set_facecolor(resolved.surface)
        for i in range(n):
            color = resolved.agent_color(i)
            ax.plot(
                positions[:, i, 0],
                positions[:, i, 1],
                color=color,
                linewidth=1.4,
                alpha=0.85,
            )
            ax.plot(
                positions[0, i, 0],
                positions[0, i, 1],
                marker="o",
                color=color,
                markersize=6,
            )
            ax.plot(
                goals[i, 0],
                goals[i, 1],
                marker="x",
                color=color,
                markersize=7,
                markeredgewidth=1.6,
            )
            ax.add_patch(
                Circle(
                    tuple(positions[-1, i]),
                    0.5,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.9,
                )
            )
        min_distance = (
            min(result.metrics.min_distance)
            if result.metrics.min_distance
            else math.nan
        )
        arrived = _arrived(result, goals)
        verdict = "arrived %d/%d" % (arrived, n)
        if arrived < n:
            verdict += " — deadlock"
        ax.set_title(
            "%s\n%s · min separation %.2f · collisions %d"
            % (label, verdict, min_distance, result.metrics.collisions),
            fontsize=9.5,
        )
        ax.set_xlim(-11.5, 11.5)
        ax.set_ylim(-11.5, 11.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    figure.suptitle(
        "circle_swap(8): the same crossing under four decentralized laws", fontsize=11
    )
    figure.tight_layout()
    path = out("navigation.png")
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    _done(started, path)


# --------------------------------------------------------------------------
# 4. lifelong MAPF: goals keep coming, throughput is the score
# --------------------------------------------------------------------------


def lifelong(out) -> None:
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    from pymapf.rl import MAPFEnv, PIBTPolicy, RandomPolicy, ReplanPolicy
    from pymapf.rl.evaluate import rollout

    resolved = theme_module.apply(THEME)

    # -- the animation: PIBT re-tasked on arrival ----------------------------
    started = _step("lifelong: PIBT in the warehouse")
    env = MAPFEnv(
        "warehouse", n_agents=8, lifelong=True, randomise=False, seed=0, max_steps=120
    )
    policy = PIBTPolicy(env)
    observations, _ = env.reset(seed=0)
    names = list(env.possible_agents)
    colors = resolved.color_map(names)
    positions = [dict(env.positions)]
    goals = [dict(env.goals)]
    completed = [0]
    while env.agents:
        observations, _, _, _, _ = env.step(policy.act(observations))
        positions.append(dict(env.positions))
        goals.append(dict(env.goals))
        completed.append(env.goals_completed)
    steps = len(positions) - 1
    substeps = 3
    frames = steps * substeps + 12

    ax = viz.plot_grid(env.grid, theme=THEME)
    figure = ax.figure
    dots = {
        name: ax.plot(
            [],
            [],
            marker="o",
            markersize=9,
            color=colors[name],
            linestyle="none",
            zorder=5,
        )[0]
        for name in names
    }
    targets = {
        name: ax.plot(
            [],
            [],
            marker="s",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=colors[name],
            markeredgewidth=1.4,
            linestyle="none",
            zorder=3,
        )[0]
        for name in names
    }
    flashes = {
        name: ax.plot(
            [],
            [],
            marker="o",
            markersize=16,
            markerfacecolor="none",
            markeredgecolor=colors[name],
            markeredgewidth=1.6,
            linestyle="none",
            zorder=6,
        )[0]
        for name in names
    }
    counter = ax.text(
        0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left", fontsize=9
    )
    figure.suptitle(
        "MAPFEnv(lifelong=True) · PIBTPolicy · a new goal the moment one is reached",
        fontsize=9.5,
        color=resolved.ink,
    )
    flash_frames = 2 * substeps

    def update(frame):
        k = min(frame // substeps, steps)
        fraction = min(1.0, (frame - k * substeps) / substeps) if k < steps else 1.0
        for name in names:
            here = np.array(positions[k][name], dtype=float)
            there = np.array(positions[min(k + 1, steps)][name], dtype=float)
            row, col = here + fraction * (there - here)
            dots[name].set_data([col], [row])
            goal = goals[k][name]
            targets[name].set_data([goal[1]], [goal[0]])
            # A goal that just changed: ring the one that was reached.
            flashed = False
            for back in range(1, flash_frames // substeps + 1):
                if k - back >= 0 and goals[k - back][name] != goals[k][name]:
                    old = goals[k - back][name]
                    flashes[name].set_data([old[1]], [old[0]])
                    flashed = True
                    break
            if not flashed:
                flashes[name].set_data([], [])
        counter.set_text(
            "step %3d   goals completed %2d   throughput %.2f / step"
            % (k, completed[k], completed[k] / max(1, k))
        )
        return (
            list(dots.values())
            + list(targets.values())
            + list(flashes.values())
            + [counter]
        )

    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 // 18, blit=True
    )
    path = _save_gif(animation, out("animated-lifelong.gif"), fps=18)
    _done(started, path)

    # -- the figure: throughput of three policies on shared instances --------
    started = _step("lifelong: throughput bars")
    theme_module.apply(THEME)
    seeds = (0, 1, 2)
    max_steps = 200
    n_agents = 8
    methods = [
        ("random", lambda e: RandomPolicy(e)),
        ("PIBT, one step at a time", lambda e: PIBTPolicy(e)),
        (
            "LaCAM, replanned on re-tasking",
            lambda e: ReplanPolicy(e, "lacam", time_limit=2.0),
        ),
    ]
    rates = []
    collisions = []
    for _label, make in methods:
        per_seed = []
        hits = 0
        for seed in seeds:
            env = MAPFEnv(
                "warehouse",
                n_agents=n_agents,
                lifelong=True,
                randomise=False,
                seed=seed,
                max_steps=max_steps,
            )
            _, summary = rollout(env, make(env), seed=seed)
            per_seed.append(summary["goals_completed"] / n_agents / max_steps * 100.0)
            hits += summary["collisions"]
        rates.append(per_seed)
        collisions.append(hits)
    figure, ax = plt.subplots(figsize=(6.8, 3.2))
    y = np.arange(len(methods))
    means = [float(np.mean(r)) for r in rates]
    spread = [float(np.std(r)) for r in rates]
    bar_colors = [resolved.axis, resolved.agent_color(0), resolved.agent_color(1)]
    ax.barh(
        y,
        means,
        xerr=spread,
        color=bar_colors,
        height=0.55,
        error_kw={"ecolor": resolved.muted},
    )
    for i, (mean, err, hits) in enumerate(zip(means, spread, collisions)):
        ax.text(
            mean + err + 0.2,
            i,
            "%.1f   (%d collisions)" % (mean, hits),
            va="center",
            fontsize=9,
            color=resolved.ink,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _ in methods], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(means) * 1.45)
    ax.set_xlabel("goals per agent per 100 steps")
    ax.set_title("Lifelong warehouse, 8 agents, 3 seeded episodes of 200 steps")
    figure.tight_layout()
    path = out("lifelong.png")
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    _done(started, path)


DEMOS = {
    "quadrotor": quadrotor,
    "3d": volume,
    "navigation": navigation,
    "lifelong": lifelong,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default=ASSETS, help="directory for the assets")
    parser.add_argument(
        "--only", nargs="*", choices=sorted(DEMOS), help="render a subset of the demos"
    )
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    def out(name: str) -> str:
        return os.path.join(args.output, name)

    started = time.perf_counter()
    for name, render in DEMOS.items():
        if args.only and name not in args.only:
            continue
        render(out)
    print("done in %.1fs" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
