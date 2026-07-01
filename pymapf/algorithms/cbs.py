"""Conflict-Based Search (CBS).

CBS is the canonical two-level optimal MAPF algorithm (Sharon et al., 2015):

* the **high level** searches a binary constraint tree, best-first on
  sum-of-costs; each node holds a set of constraints per agent;
* the **low level** replans a single agent under its constraints using the
  shared :func:`~pymapf.algorithms.space_time_astar.space_time_astar`.

On each expansion we find the first conflict in the current joint plan and
branch by forbidding one of the two involved agents from that (vertex or edge)
constraint, then replan only that agent. The first conflict-free node is
returned and is optimal in sum-of-costs.
"""

from __future__ import annotations

import heapq
from itertools import count
from typing import Dict, List, Optional

from ..core.solver import (
    Conflict,
    Constraints,
    MAPFProblem,
    MAPFSolver,
    Solution,
    find_first_conflict,
    register_solver,
)
from .space_time_astar import space_time_astar


class _Node:
    __slots__ = ("constraints", "paths", "cost")

    def __init__(self, constraints, paths):
        self.constraints: Dict[str, Constraints] = constraints
        self.paths: Dict[str, List] = paths
        self.cost = sum(len(p) - 1 for p in paths.values())


@register_solver("cbs")
class ConflictBasedSearch(MAPFSolver):
    """Optimal (sum-of-costs) conflict-based search.

    Args:
        heuristic: name or callable for the low-level search.
        max_expansions: safety cap on high-level nodes; returns ``None`` if
            exceeded (guards against pathological/unsolvable instances).
    """

    def __init__(self, heuristic="manhattan", max_expansions: int = 10000):
        self.heuristic = heuristic
        self.max_expansions = max_expansions

    def solve(self, problem: MAPFProblem) -> Optional[Solution]:
        agents = list(problem.agents)

        # Root: plan every agent independently, with no constraints.
        root_constraints = {a.name: Constraints() for a in agents}
        root_paths = {}
        for a in agents:
            path = self._low_level(problem, a.start, a.goal, root_constraints[a.name])
            if path is None:
                return None
            root_paths[a.name] = path

        counter = count()
        open_heap = [(0, next(counter), _Node(root_constraints, root_paths))]
        expansions = 0

        while open_heap and expansions < self.max_expansions:
            _, _, node = heapq.heappop(open_heap)
            expansions += 1

            conflict = find_first_conflict(node.paths)
            if conflict is None:
                return Solution(
                    paths=node.paths, algorithm=self.name, expansions=expansions
                )

            for agent_name in self._agents_to_constrain(conflict):
                child = self._branch(problem, node, conflict, agent_name)
                if child is not None:
                    heapq.heappush(open_heap, (child.cost, next(counter), child))

        return None

    @staticmethod
    def _agents_to_constrain(conflict: Conflict):
        return (conflict.a, conflict.b)

    def _branch(
        self, problem: MAPFProblem, node: _Node, conflict: Conflict, agent_name: str
    ) -> Optional[_Node]:
        """Create the child node that forbids ``agent_name`` from the conflict."""
        constraints = {n: c.copy() for n, c in node.constraints.items()}
        ac = constraints[agent_name]
        if conflict.kind == "vertex":
            ac.add_vertex(conflict.cell_a, conflict.t)
        else:
            # Edge conflict: agent `a` traversed cell_b -> cell_a and `b`
            # traversed cell_a -> cell_b, both arriving at conflict.t.
            if agent_name == conflict.a:
                ac.add_edge(conflict.cell_b, conflict.cell_a, conflict.t)
            else:
                ac.add_edge(conflict.cell_a, conflict.cell_b, conflict.t)

        agent = next(a for a in problem.agents if a.name == agent_name)
        path = self._low_level(problem, agent.start, agent.goal, ac)
        if path is None:
            return None
        paths = dict(node.paths)
        paths[agent_name] = path
        return _Node(constraints, paths)

    def _low_level(self, problem, start, goal, constraints):
        return space_time_astar(
            problem.grid,
            start,
            goal,
            constraints=constraints,
            heuristic=self.heuristic,
            allow_diagonals=problem.allow_diagonals,
        )
