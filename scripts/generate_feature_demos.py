"""Render the demos for the kinodynamic, 3D, navigation and lifelong layers.

One animation and one figure per feature, written to ``.docs/assets`` in the
same theme as the rest of the gallery, so the README and the site show what
each layer does rather than describe it::

    python scripts/generate_feature_demos.py             # everything
    python scripts/generate_feature_demos.py --only orca  # one of them

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
# 1. kinodynamic: a discrete plan executed under real speed and acceleration
# --------------------------------------------------------------------------


def kinodynamic(out) -> None:
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    from pymapf.kinodynamic import KinematicLimits, plan_trajectories

    resolved = theme_module.apply(THEME)
    scenario = pymapf.build_scenario("warehouse", n_agents=8, seed=0)
    solution = _solve(scenario.to_problem())
    limits = KinematicLimits(v_max=1.5, a_max=3.0, safety_distance=0.5)
    trajectories = plan_trajectories(solution, limits=limits)
    names = list(solution.paths)
    colors = resolved.color_map(names)
    makespan = trajectories.makespan
    separation = trajectories.min_separation()[0]

    # -- the animation: agents moving at their real speed --------------------
    started = _step("kinodynamic: animating the warehouse plan")
    dt = 1.0 / 12.0
    frames = int(math.ceil(makespan / dt)) + 18  # hold the final state
    ax = viz.plot_grid(scenario, theme=THEME)
    figure = ax.figure
    for agent in scenario.agents:
        ax.plot(
            agent.goal[1],
            agent.goal[0],
            marker="s",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=colors[agent.name],
            markeredgewidth=1.4,
            zorder=3,
        )
    dots = {
        name: ax.plot(
            [],
            [],
            marker="o",
            markersize=10,
            color=colors[name],
            linestyle="none",
            zorder=5,
        )[0]
        for name in names
    }
    trails = {
        name: ax.plot([], [], color=colors[name], linewidth=2.2, alpha=0.55, zorder=4)[
            0
        ]
        for name in names
    }
    clock = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=resolved.ink,
        fontsize=9,
    )
    figure.suptitle(
        "plan_trajectories() — v_max %.1f, a_max %.1f, safety %.1f · min separation %.2f"
        % (limits.v_max, limits.a_max, limits.safety_distance, separation),
        fontsize=10,
        color=resolved.ink,
    )
    trail_seconds = 1.6

    def update(frame):
        t = min(frame * dt, makespan)
        for name in names:
            trajectory = trajectories[name]
            row, col = trajectory.position_at(t)
            dots[name].set_data([col], [row])
            past = np.linspace(max(0.0, t - trail_seconds), t, 12)
            points = np.array([trajectory.position_at(s) for s in past])
            trails[name].set_data(points[:, 1], points[:, 0])
        speed = max(math.hypot(*trajectories[name].velocity_at(t)) for name in names)
        clock.set_text("t = %5.2f s   fastest agent %.2f m/s" % (t, speed))
        return list(dots.values()) + list(trails.values()) + [clock]

    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 * dt, blit=True
    )
    path = _save_gif(animation, out("animated-kinodynamic.gif"), fps=12)
    _done(started, path)

    # -- the figure: every agent's speed over time ---------------------------
    started = _step("kinodynamic: speed profiles")
    theme_module.apply(THEME)
    figure, axes = plt.subplots(
        2, 1, figsize=(7.2, 4.6), sharex=True, height_ratios=[3, 2]
    )
    times = np.linspace(0.0, makespan, 900)
    highlighted = names[:3]
    for name in names:
        trajectory = trajectories[name]
        speeds = [math.hypot(*trajectory.velocity_at(t)) for t in times]
        if name in highlighted:
            axes[0].plot(
                times, speeds, color=colors[name], linewidth=1.7, label=name, zorder=3
            )
        else:
            axes[0].plot(
                times, speeds, color=colors[name], linewidth=0.9, alpha=0.28, zorder=2
            )
    axes[0].axhline(limits.v_max, color=resolved.muted, linewidth=0.8, linestyle="--")
    axes[0].text(
        makespan, limits.v_max, " v_max", color=resolved.muted, va="center", fontsize=8
    )
    axes[0].set_ylabel("speed (m/s)")
    axes[0].set_title(
        "Speed: a rest-to-rest trapezoid per move, capped at v_max (three agents highlighted)",
        fontsize=10,
    )
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")

    # the discrete plan's timing against the scheduled one, per agent
    discrete = [len(solution.paths[name]) - 1 for name in names]
    scheduled = [trajectories[name].arrival_time for name in names]
    y = np.arange(len(names))
    axes[1].barh(
        y - 0.18, discrete, height=0.36, color=resolved.axis, label="discrete steps"
    )
    axes[1].barh(
        y + 0.18,
        scheduled,
        height=0.36,
        color=[colors[n] for n in names],
        label="seconds",
    )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("timesteps / seconds")
    axes[1].set_title(
        "Arrival time: discrete steps at unit speed vs. seconds under the limits",
        fontsize=10,
    )
    axes[1].legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    path = out("kinodynamic-profiles.png")
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
    "kinodynamic": kinodynamic,
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
