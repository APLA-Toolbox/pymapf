"""Planner-backed policies that step the environment like a learned one would.

A learned policy sees observations; a planner sees the whole state. These
classes give a planner the policy interface -- ``act(observations) ->
{agent: action}`` -- by reading the environment they are bound to, so the
evaluation harness can run a planner and a network through the very same
loop and score them on the very same episodes. That is the whole point of
this layer: the baseline is not a different program measured differently.

Three of them:

* :class:`PIBTPolicy` re-decides every step from the current positions and
  goals with :func:`~pymapf.algorithms.pibt.pibt_step`, with the priority rule
  from the PIBT paper's lifelong experiments -- an agent's priority grows
  every step it has not reached its goal and resets when it does. It needs no
  plan and no replanning, which is why it is the standard lifelong baseline.
* :class:`ReplanPolicy` runs a full solver from the current configuration
  whenever any goal changes, and executes the plan until the next change.
  Expensive, and complete when the solver is.
* :class:`RandomPolicy` is the floor.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np

from ..core.grid import Cell

__all__ = ["PIBTPolicy", "ReplanPolicy", "RandomPolicy"]


class _EnvPolicy:
    """Common ground: bound to one env, converts moves to action indices."""

    def __init__(self, env):
        self.env = env
        self._deltas = {delta: index for index, delta in enumerate(env.actions)}

    def _action_for(self, agent: str, target: Cell) -> int:
        current = self.env.positions[agent]
        delta = tuple(t - c for t, c in zip(target, current))
        return self._deltas.get(delta, 0)

    def act(self, observations, deterministic: bool = True) -> Dict[str, int]:
        raise NotImplementedError


class RandomPolicy(_EnvPolicy):
    """Uniform random actions, seeded."""

    def __init__(self, env, seed: int = 0):
        super().__init__(env)
        self._random = np.random.default_rng(seed)

    def act(self, observations, deterministic: bool = True) -> Dict[str, int]:
        n = len(self.env.actions)
        return {agent: int(self._random.integers(n)) for agent in observations}


class PIBTPolicy(_EnvPolicy):
    """One PIBT step per environment step, from the live state.

    Distance tables are built per goal and cached, so a goal that comes back
    (which lifelong instances do, on a small map) costs nothing the second
    time. Priorities follow the paper's lifelong rule.
    """

    def __init__(self, env, seed: int = 0):
        super().__init__(env)
        self._tables: Dict[Cell, dict] = {}
        self._elapsed: Dict[str, int] = {}
        self._rng = random.Random(seed)

    def _table(self, goal: Cell) -> dict:
        table = self._tables.get(goal)
        if table is None:
            from ..algorithms.search import distance_table

            table = distance_table(self.env.grid, goal, self.env.allow_diagonals)
            self._tables[goal] = table
        return table

    def act(self, observations, deterministic: bool = True) -> Dict[str, int]:
        from ..algorithms.pibt import pibt_step

        env = self.env
        names = list(env.possible_agents)
        if env.step_count == 0:
            self._elapsed = {name: 0 for name in names}
            # A new instance may have a new grid; tables belong to the old one.
            self._tables = {}

        def distance(agent: str, cell: Cell) -> float:
            return float(self._table(env.goals[agent]).get(cell, float("inf")))

        priorities = {
            name: self._elapsed.get(name, 0) + self._rng.random() * 1e-3
            for name in names
        }
        nxt = pibt_step(
            env.grid,
            names,
            dict(env.positions),
            dict(env.goals),
            priorities,
            distance,
            allow_diagonals=env.allow_diagonals,
            rng=self._rng,
        )
        if nxt is None:  # cannot happen without a forced assignment; stay put
            nxt = dict(env.positions)
        for name in names:
            arrived = nxt[name] == env.goals[name]
            self._elapsed[name] = 0 if arrived else self._elapsed.get(name, 0) + 1
        return {
            name: self._action_for(name, nxt[name])
            for name in names
            if name in observations
        }


class ReplanPolicy(_EnvPolicy):
    """Replan with a registered solver whenever a goal changes, then follow it.

    Args:
        algorithm: a solver name, ``"lacam"`` by default -- complete and fast,
            which matters when the plan is rebuilt many times per episode.
        time_limit: per replan. A failed replan makes every agent wait one
            step and try again from wherever it then is.
    """

    def __init__(self, env, algorithm: str = "lacam", time_limit: float = 2.0):
        super().__init__(env)
        self.algorithm = algorithm
        self.time_limit = time_limit
        self._plan: Optional[Dict[str, List[Cell]]] = None
        self._cursor = 0
        self._goals_planned: Optional[tuple] = None
        self.replans = 0

    def _replan(self) -> None:
        from ..core.solver import Agent, MAPFProblem
        from .evaluate import plan

        env = self.env
        problem = MAPFProblem(
            env.grid,
            [
                Agent(name, env.positions[name], env.goals[name])
                for name in env.possible_agents
            ],
            allow_diagonals=env.allow_diagonals,
        )
        solution = plan(problem, self.algorithm, self.time_limit)
        self.replans += 1
        self._plan = None if solution is None else dict(solution.paths)
        self._cursor = 0
        self._goals_planned = tuple(env.goals[name] for name in env.possible_agents)

    def act(self, observations, deterministic: bool = True) -> Dict[str, int]:
        env = self.env
        current_goals = tuple(env.goals[name] for name in env.possible_agents)
        if (
            env.step_count == 0
            or self._plan is None
            or current_goals != self._goals_planned
        ):
            self._replan()
        actions = {}
        self._cursor += 1
        for name in env.possible_agents:
            if name not in observations:
                continue
            if self._plan is None:
                actions[name] = 0
                continue
            path = self._plan[name]
            target = path[min(self._cursor, len(path) - 1)]
            actions[name] = self._action_for(name, target)
        return actions
