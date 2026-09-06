"""Decentralized navigation: every agent has its own goal and nobody plans for anyone else.

The flocking and formation behaviors in this package share one waypoint. The
methods here give every agent a destination of its own and ask it to get there
without hitting anyone, from local information only -- the problem a fleet
faces once the central planner is gone, or was never there.

Four control laws, spanning thirty years of the literature and two different
ideas of what "safe" means:

* :class:`ORCA` (van den Berg et al. 2011) and :class:`BufferedVoronoi`
  (Zhou et al. 2017) are **constraint** methods: each agent computes a convex
  set of velocities (or positions) that are provably safe if everyone plays by
  the same rule, and picks the one closest to what it wanted.
* :class:`PotentialField` (Khatib 1986) and :class:`SocialForce` (Helbing and
  Molnár 1995) are **force** methods: attraction to the goal, repulsion from
  everything else, summed. Simple, tunable, and with no guarantee at all --
  which the tests demonstrate rather than hide.

All four are :class:`~pymapf.swarm.base.Behavior` subclasses and register by
name, so a navigation law is swapped for a flocking one by a string, they run
in the same :class:`~pymapf.swarm.simulator.SwarmSimulator`, and the same
safety metrics apply. All are written for any dimension: ORCA's velocity
obstacle is rotationally symmetric about the line between two agents, so its
geometry lives in a plane whatever the ambient dimension, and the projection
that selects a velocity is dimension-agnostic by construction.

    from pymapf.swarm import SwarmSimulator, circle_swap

    state, goals = circle_swap(n=12, radius=10.0)
    result = SwarmSimulator("orca", initial=state, goals=goals).run(steps=400)
    result.metrics.collisions      # 0
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import Behavior, SwarmState, limit, register_behavior

__all__ = [
    "NavigationBehavior",
    "ORCA",
    "BufferedVoronoi",
    "PotentialField",
    "SocialForce",
    "circle_swap",
    "project_onto_polytope",
]


# --------------------------------------------------------------------------
# geometry shared by the constraint methods
# --------------------------------------------------------------------------


_EPS = 1e-9


def _orthonormal_complement(normal: np.ndarray) -> np.ndarray:
    """Columns spanning the hyperplane orthogonal to ``normal`` (a unit vector)."""
    d = normal.shape[0]
    # Householder reflection mapping e_1 to normal: its remaining columns span
    # the complement exactly, with no Gram-Schmidt drift.
    e1 = np.zeros(d)
    e1[0] = 1.0
    v = normal - e1 if normal[0] < 0 else normal + e1
    v_norm = np.linalg.norm(v)
    if v_norm < _EPS:
        return np.eye(d)[:, 1:]
    v = v / v_norm
    H = np.eye(d) - 2.0 * np.outer(v, v)
    return H[:, 1:]


def project_onto_polytope(
    target: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    radius: Optional[float] = None,
) -> Tuple[np.ndarray, bool]:
    """The point closest to ``target`` with ``normals @ x >= offsets`` and
    ``|x| <= radius`` (when given).

    Incremental, after the linear programs of RVO2 (van den Berg et al. 2011)
    and Seidel (1991): start from the target (clamped to the ball) and add the
    half-spaces one at a time. If the current optimum satisfies the new one,
    nothing changes. If it violates it, the new optimum must lie *on* that
    hyperplane -- the objective is strictly convex -- so the problem is solved
    again inside the hyperplane, one dimension lower, with the constraints
    seen so far. The recursion bottoms out on a line, where every constraint
    is an interval and the answer is a clamp.

    Exact, deterministic, and cheap for the handful of dimensions and dozens
    of constraints a navigation law produces. Returns ``(point, feasible)``;
    on an empty intersection the point is the last feasible iterate and the
    caller decides how to relax.
    """
    target = np.asarray(target, dtype=float)
    normals = np.asarray(normals, dtype=float).reshape(-1, target.shape[0])
    offsets = np.asarray(offsets, dtype=float).reshape(-1)
    return _solve(target, normals, offsets, radius)


def _solve(target, normals, offsets, radius):
    d = target.shape[0]
    if d == 1:
        return _solve_line(target, normals, offsets, radius)

    if radius is not None:
        norm = float(np.linalg.norm(target))
        x = target if norm <= radius else target * (radius / norm)
    else:
        x = target.copy()

    for i in range(normals.shape[0]):
        n_i, b_i = normals[i], offsets[i]
        if float(np.dot(n_i, x)) >= b_i - _EPS:
            continue
        # Restrict to the hyperplane n_i . x = b_i. With unit n_i the closest
        # point of the plane to the origin is b_i n_i, and the plane is
        # p0 + Q y for an orthonormal basis Q of the complement.
        n_norm = float(np.linalg.norm(n_i))
        if n_norm < _EPS:
            if b_i > _EPS:
                return x, False
            continue
        unit = n_i / n_norm
        b_unit = b_i / n_norm
        p0 = b_unit * unit
        Q = _orthonormal_complement(unit)

        sub_radius = None
        if radius is not None:
            if abs(b_unit) > radius + _EPS:
                return x, False  # the plane misses the ball entirely
            sub_radius = float(np.sqrt(max(radius * radius - b_unit * b_unit, 0.0)))

        sub_target = Q.T @ (target - p0)
        sub_normals = normals[:i] @ Q
        sub_offsets = offsets[:i] - normals[:i] @ p0
        y, ok = _solve(sub_target, sub_normals, sub_offsets, sub_radius)
        if not ok:
            return x, False
        x = p0 + Q @ y
    return x, True


def _solve_line(target, normals, offsets, radius):
    """One-dimensional: every half-space is a half-line, the ball an interval."""
    lo, hi = -np.inf, np.inf
    if radius is not None:
        lo, hi = -radius, radius
    for a, b in zip(normals[:, 0], offsets):
        if abs(a) < _EPS:
            if b > _EPS:
                return np.array([0.0]), False
            continue
        bound = b / a
        if a > 0:
            lo = max(lo, bound)
        else:
            hi = min(hi, bound)
    if lo > hi + _EPS:
        return np.array([0.5 * (lo + hi)]), False
    t = float(target[0])
    return np.array([min(max(t, lo), hi)]), True


def _perpendicular_to(axis: np.ndarray, hint: np.ndarray) -> np.ndarray:
    """A unit vector orthogonal to ``axis``, on ``hint``'s side when it has one."""
    perp = hint - np.dot(hint, axis) * axis
    norm = float(np.linalg.norm(perp))
    if norm > 1e-9:
        return perp / norm
    # ``hint`` is collinear with the axis: any perpendicular will do, chosen
    # deterministically from the basis vector least aligned with the axis.
    basis = np.zeros_like(axis)
    basis[int(np.argmin(np.abs(axis)))] = 1.0
    perp = basis - np.dot(basis, axis) * axis
    return perp / float(np.linalg.norm(perp))


# --------------------------------------------------------------------------
# the shared base: goals, preferred velocity, arrival
# --------------------------------------------------------------------------


class NavigationBehavior(Behavior):
    """A behavior whose agents each have a goal of their own.

    Args:
        goals: ``(n, d)`` destinations, one per agent. ``None`` sends everyone
            to ``params.migration_point``, so a navigation law still runs in a
            simulator configured for flocking.
        radius: the agent's physical radius. Defaults to half of
            ``params.separation_distance``, so "collision" means the same thing
            to these behaviors as it does to the swarm metrics.
        goal_tolerance: distance at which an agent counts as arrived and stops.
        slow_radius: distance from the goal at which the preferred speed
            starts ramping down, so agents settle rather than orbit.

    The output is a **velocity** the integrator tracks directly (as with the
    kinematic flocking models), except for :class:`SocialForce`, which is a
    force model and returns an acceleration.
    """

    output = "velocity"

    def __init__(
        self,
        goals: Optional[Sequence[Sequence[float]]] = None,
        radius: Optional[float] = None,
        goal_tolerance: float = 0.3,
        slow_radius: Optional[float] = None,
        clearance: float = 0.01,
        tie_break: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._goals = (
            None if goals is None else np.atleast_2d(np.asarray(goals, dtype=float))
        )
        self._radius = radius
        self.goal_tolerance = goal_tolerance
        self._slow_radius = slow_radius
        #: Magnitude of a seeded random perturbation added to the preferred
        #: velocity each step. Exactly symmetric encounters -- agents on a
        #: ring, all facing the centre -- give every agent a perfect tie
        #: between going left and going right, and a deterministic rule
        #: resolves it identically for all of them, forever. RVO2's own circle
        #: demo perturbs the preferred velocity by 1e-4 for this reason.
        self.tie_break = tie_break
        self._rng = np.random.default_rng(0)
        #: Added to the radius when constraints are built, so that two agents
        #: whose constraints are exactly tight sit a hair *outside* the
        #: separation distance rather than on it, where floating point decides
        #: whether the metrics call it a collision.
        self.clearance = clearance
        self.goals: Optional[np.ndarray] = None

    # -- configuration --------------------------------------------------
    @property
    def radius(self) -> float:
        base = (
            self._radius
            if self._radius is not None
            else self.params.separation_distance / 2
        )
        return base + self.clearance

    @property
    def slow_radius(self) -> float:
        return self._slow_radius if self._slow_radius is not None else 3 * self.radius

    def reset(self, state: SwarmState) -> None:
        if self._goals is not None:
            goals = self._goals
        elif self.params.migration_point is not None:
            goals = np.tile(
                np.asarray(self.params.migration_point, dtype=float), (state.n, 1)
            )
        else:
            goals = state.positions.copy()  # nowhere to go: hold position
        if goals.shape != state.positions.shape:
            raise ValueError(
                "goals %s do not match the swarm's %s"
                % (goals.shape, state.positions.shape)
            )
        self.goals = goals
        self._rng = np.random.default_rng(self.params.seed)

    def _ensure_goals(self, state: SwarmState) -> np.ndarray:
        if self.goals is None or self.goals.shape != state.positions.shape:
            self.reset(state)
        return self.goals

    # -- the parts every law shares ------------------------------------
    def goal(self, state: SwarmState, index: int) -> np.ndarray:
        return self._ensure_goals(state)[index]

    def to_goal(self, state: SwarmState, index: int) -> np.ndarray:
        return self.goal(state, index) - state.positions[index]

    def preferred_velocity(self, state: SwarmState, index: int) -> np.ndarray:
        """Straight at the goal at cruise speed, ramping down to stop on it."""
        offset = self.to_goal(state, index)
        distance = float(np.linalg.norm(offset))
        if distance <= self.goal_tolerance:
            return np.zeros(state.dimension)
        speed = self.params.cruise_speed * min(1.0, distance / self.slow_radius)
        preferred = offset * (speed / distance)
        if self.tie_break > 0:
            preferred = preferred + self._rng.normal(
                0.0, self.tie_break, size=state.dimension
            )
        return preferred

    def distances_to_goal(self, state: SwarmState) -> np.ndarray:
        return np.linalg.norm(self._ensure_goals(state) - state.positions, axis=1)

    def arrived(self, state: SwarmState) -> np.ndarray:
        return self.distances_to_goal(state) <= self.goal_tolerance

    def obstacles(self) -> List[Tuple[np.ndarray, float]]:
        return [
            (np.asarray(c, dtype=float), float(r)) for c, r in self.params.obstacles
        ]

    def __repr__(self) -> str:
        goals = "none" if self.goals is None else "%dx%d" % self.goals.shape
        return "%s(name=%r, goals=%s, radius=%.2f)" % (
            type(self).__name__,
            self.name,
            goals,
            self.radius,
        )


# --------------------------------------------------------------------------
# ORCA
# --------------------------------------------------------------------------


@register_behavior("orca")
class ORCA(NavigationBehavior):
    """Optimal Reciprocal Collision Avoidance (van den Berg et al. 2011).

    For each neighbour, the set of relative velocities that lead to a
    collision within ``time_horizon`` is a truncated cone -- the velocity
    obstacle. ORCA takes the smallest change ``u`` to the current relative
    velocity that leaves the cone, has each agent take **half** of it (the
    reciprocity: both see the same geometry and split the work, so neither
    over-reacts), and turns that into a half-space of permitted velocities.
    The intersection of all half-spaces with the speed disc is convex, and
    the velocity chosen is the one in it closest to the preferred velocity.

    Static obstacles are the same construction with the agent taking all of
    ``u`` (an obstacle will not do its half) and its own, shorter
    ``obstacle_horizon``.

    The velocity is selected by :func:`project_onto_polytope`, the same
    incremental construction as RVO2's linear programs generalised to any
    dimension by recursion, so the answer is exact. One deviation from RVO2,
    stated so nobody has to diff the source: when the constraint set is
    empty -- a crowd too dense for the horizon -- RVO2's third program
    minimises the worst violation of the agent half-planes; here they are
    relaxed by a common margin found by bisection until the set is
    non-empty, the same objective met less exactly. Obstacle constraints
    are never relaxed.

    The geometry is written for any dimension. The velocity obstacle is
    rotationally symmetric about the line between two agents, so the leg
    projection happens in the plane spanned by that line and the relative
    velocity, whatever the ambient dimension.
    """

    name = "orca"

    def __init__(
        self,
        time_horizon: float = 4.0,
        obstacle_horizon: float = 2.0,
        dt: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.time_horizon = time_horizon
        self.obstacle_horizon = obstacle_horizon
        self.dt = dt

    # -- one half-space -------------------------------------------------
    def _avoidance(
        self,
        rel_pos: np.ndarray,
        rel_vel: np.ndarray,
        combined_radius: float,
        horizon: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The smallest change ``u`` taking ``rel_vel`` out of the velocity
        obstacle, and the outward normal of the resulting half-space."""
        dist_sq = float(np.dot(rel_pos, rel_pos))
        radius_sq = combined_radius * combined_radius
        if dist_sq > radius_sq:
            # Not yet colliding: the truncated cone.
            w = rel_vel - rel_pos / horizon
            w_len_sq = float(np.dot(w, w))
            dot = float(np.dot(w, rel_pos))
            if dot < 0.0 and dot * dot > radius_sq * w_len_sq:
                # Project onto the cut-off circle.
                w_len = math.sqrt(w_len_sq)
                unit_w = w / max(w_len, 1e-12)
                u = (combined_radius / horizon - w_len) * unit_w
                return u, unit_w
            # Project onto a leg of the cone, in the plane of rel_pos and rel_vel.
            dist = math.sqrt(dist_sq)
            leg = math.sqrt(max(dist_sq - radius_sq, 0.0))
            axis = rel_pos / dist
            side = _perpendicular_to(axis, rel_vel)
            # Unit direction of the leg on rel_vel's side of the axis, and the
            # leg's *outward* normal -- perpendicular to it, away from the
            # cone's interior. The normal is the half-space's, whichever way
            # ``u`` points: when the relative velocity is already outside the
            # cone ``u`` points inward (the agent may drift toward the cone),
            # and taking ``u``'s direction as the normal would flip the
            # permitted side. That bug let two agents brush past at 0.985.
            direction = (axis * leg + side * combined_radius) / dist
            normal = (side * leg - axis * combined_radius) / dist
            u = float(np.dot(rel_vel, direction)) * direction - rel_vel
            return u, normal
        # Already overlapping: get out within one step.
        w = rel_vel - rel_pos / self.dt
        w_len = float(np.linalg.norm(w))
        unit_w = w / max(w_len, 1e-12)
        u = (combined_radius / self.dt - w_len) * unit_w
        return u, unit_w

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        me = state.positions[index]
        my_velocity = state.velocities[index]
        radius = self.radius
        agent_normals: List[np.ndarray] = []
        agent_points: List[np.ndarray] = []
        for j in self.neighbors(state, index):
            rel_pos = state.positions[j] - me
            rel_vel = my_velocity - state.velocities[j]
            u, normal = self._avoidance(rel_pos, rel_vel, 2 * radius, self.time_horizon)
            agent_normals.append(normal)
            agent_points.append(my_velocity + 0.5 * u)

        obstacle_normals: List[np.ndarray] = []
        obstacle_points: List[np.ndarray] = []
        for centre, obstacle_radius in self.obstacles():
            rel_pos = centre - me
            if (
                float(np.linalg.norm(rel_pos))
                > self.params.sensing_range + obstacle_radius
            ):
                continue
            u, normal = self._avoidance(
                rel_pos, my_velocity, radius + obstacle_radius, self.obstacle_horizon
            )
            obstacle_normals.append(normal)
            obstacle_points.append(my_velocity + u)

        preferred = self.preferred_velocity(state, index)
        d = state.dimension
        normals = np.array(agent_normals + obstacle_normals).reshape(-1, d)
        points = np.array(agent_points + obstacle_points).reshape(-1, d)
        n_agents = len(agent_normals)
        # (v - point) . normal >= 0  <=>  normal . v >= normal . point
        base_offsets = np.einsum("ij,ij->i", normals, points)

        def solve(margin: float):
            offsets = base_offsets.copy()
            offsets[:n_agents] -= margin  # relax the reciprocal planes only
            return project_onto_polytope(
                preferred, normals, offsets, self.params.max_speed
            )

        velocity, feasible = solve(0.0)
        if not feasible:
            # A crowd too dense for the horizon: relax the reciprocal
            # half-planes by a common margin, found by bisection, until the
            # set is non-empty -- RVO2's third program's objective, met less
            # exactly. Obstacle planes are never relaxed.
            lo, hi = 0.0, 2.0 * self.params.max_speed
            velocity = np.zeros(d)
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                candidate, ok = solve(mid)
                if ok:
                    velocity, hi = candidate, mid
                else:
                    lo = mid
        return velocity


# --------------------------------------------------------------------------
# buffered Voronoi cells
# --------------------------------------------------------------------------


@register_behavior("buffered_voronoi")
class BufferedVoronoi(NavigationBehavior):
    """Buffered Voronoi cells (Zhou, Wang, Bandyopadhyay and Schwager 2017).

    Each agent owns the Voronoi cell of its position, shrunk inward by the
    safety radius. Cells are disjoint by construction, so as long as every
    agent stays inside its own cell for the step, no two can touch --
    without any velocity information about the others at all, which is what
    makes the method attractive with cheap sensing. Each step the agent moves
    toward the point of its cell closest to its goal.

    The cell is the intersection of half-spaces, one per neighbour, and the
    closest point of a convex polytope is found by
    :func:`project_onto_polytope`. The agent's own position is always in
    its cell, so the segment from it to the target is too, and capping the
    speed keeps the next position on that segment.

    Known limitation, from the paper: symmetric configurations can deadlock,
    with agents parked on cell boundaries facing each other. The paper's
    remedy is a right-hand detour -- when the cell's closest point to the
    goal is the agent's own position, aim beside the goal instead -- and it
    is implemented as ``detour``; a test runs with it off so the deadlock is
    on record.

    Static circular obstacles enter as half-spaces tangent to the obstacle
    inflated by the safety radius -- conservative, since the true cell would
    be bounded by a curve.
    """

    name = "buffered_voronoi"

    def __init__(
        self,
        dt: float = 0.1,
        gain: float = 1.0,
        detour: bool = True,
        detour_steps: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.gain = gain
        self.detour_steps = detour_steps
        self._detouring: dict = {}
        #: When the closest point of the cell to the goal is the agent's own
        #: position and the goal is still far, the agent is wedged against a
        #: neighbour. The paper's remedy is a sidestep: aim at a point beside
        #: the goal direction instead, and the cell's closest point moves.
        self.detour = detour

    @staticmethod
    def _detour_hint(dimension: int) -> np.ndarray:
        # A fixed "right-hand" side: the last axis, which is a right-hand turn
        # from any heading in the plane and an arbitrary but consistent side
        # in a volume.
        hint = np.zeros(dimension)
        hint[-1] = 1.0
        return hint

    def reset(self, state: SwarmState) -> None:
        super().reset(state)
        self._detouring = {}

    def cell(self, state: SwarmState, index: int):
        """The buffered Voronoi cell of one agent as ``(normals, offsets)`` with
        the cell being ``normals @ x >= offsets``."""
        me = state.positions[index]
        safety = self.radius
        normals: List[np.ndarray] = []
        offsets: List[float] = []
        for j in self.neighbors(state, index):
            offset = state.positions[j] - me
            distance = float(np.linalg.norm(offset))
            if distance < 1e-9:
                continue
            unit = offset / distance
            # Boundary at the midpoint, moved a safety radius toward me; the
            # cell is on my side of it.
            boundary = me + unit * (0.5 * distance - safety)
            normals.append(-unit)
            offsets.append(float(np.dot(-unit, boundary)))
        for centre, obstacle_radius in self.obstacles():
            offset = centre - me
            distance = float(np.linalg.norm(offset))
            if (
                distance < 1e-9
                or distance > self.params.sensing_range + obstacle_radius
            ):
                continue
            unit = offset / distance
            boundary = centre - unit * (obstacle_radius + safety)
            normals.append(-unit)
            offsets.append(float(np.dot(-unit, boundary)))
        d = state.dimension
        return np.array(normals).reshape(-1, d), np.array(offsets)

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        me = state.positions[index]
        goal = self.goal(state, index)
        normals, offsets = self.cell(state, index)
        target, feasible = project_onto_polytope(goal, normals, offsets)
        if not feasible:
            # Only possible once separation is already lost; hold still rather
            # than move on a cell that does not exist.
            return np.zeros(state.dimension)
        to_goal = goal - me
        remaining = float(np.linalg.norm(to_goal))
        if remaining <= self.goal_tolerance:
            return np.zeros(state.dimension)
        offset = target - me
        distance = float(np.linalg.norm(offset))
        creeping = distance < 0.05 * self.params.cruise_speed * self.dt
        if self.detour and (creeping or self._detouring.get(index, 0) > 0):
            # Wedged, or recently so: sidestep. Aim beside the goal, on a
            # fixed side so the choice is consistent step to step, and take
            # the cell's closest point to that instead. The sidestep persists
            # for a few steps so the agent clears the neighbour rather than
            # flip-flopping on the boundary.
            if creeping:
                self._detouring[index] = self.detour_steps
            self._detouring[index] -= 1
            side = _perpendicular_to(
                to_goal / remaining, self._detour_hint(state.dimension)
            )
            detour_goal = me + side * self.slow_radius + to_goal * 0.5
            target, feasible = project_onto_polytope(detour_goal, normals, offsets)
            offset = target - me
            distance = float(np.linalg.norm(offset))
        if distance <= 1e-9:
            return np.zeros(state.dimension)
        # Reach the target this step if allowed; never overshoot it, so the
        # next position stays on the segment inside the cell.
        speed = min(self.params.cruise_speed, self.gain * distance / self.dt)
        return offset * (speed / distance)


# --------------------------------------------------------------------------
# potential fields
# --------------------------------------------------------------------------


@register_behavior("potential_field")
class PotentialField(NavigationBehavior):
    """Artificial potential fields (Khatib 1986).

    The goal is the bottom of a bowl, every other agent and obstacle the tip
    of a spike, and the agent rolls downhill: velocity proportional to the
    negative gradient. Attraction is linear in distance (capped), repulsion is
    Khatib's ``k (1/d - 1/rho_0) / d^2`` inside an influence distance
    ``rho_0`` and zero outside.

    No guarantee of anything. The sum of a bowl and a spike has a local
    minimum on the far side of the spike, and an agent whose goal is directly
    behind an obstacle stops there -- the well-known failure, reproduced in
    the tests. Two agents approaching head-on likewise stall on the line
    between them until numerical asymmetry breaks the tie. It is here as the
    baseline every later method was measured against.
    """

    name = "potential_field"

    def __init__(
        self,
        attraction: float = 1.0,
        repulsion: float = 2.0,
        influence: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attraction = attraction
        self.repulsion = repulsion
        self._influence = influence

    @property
    def influence(self) -> float:
        return self._influence if self._influence is not None else 4 * self.radius

    def _repulsion(self, offset: np.ndarray, clearance: float) -> np.ndarray:
        """Khatib's spike, for a surface ``clearance`` away along ``offset``."""
        distance = float(np.linalg.norm(offset))
        if distance < 1e-9:
            return np.zeros_like(offset)
        rho = max(distance - clearance, 1e-3)
        if rho >= self.influence:
            return np.zeros_like(offset)
        magnitude = self.repulsion * (1.0 / rho - 1.0 / self.influence) / (rho * rho)
        return magnitude * offset / distance

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        me = state.positions[index]
        if float(np.linalg.norm(self.goal(state, index) - me)) <= self.goal_tolerance:
            return np.zeros(state.dimension)
        force = self.attraction * self.preferred_velocity(state, index)
        for j in self.neighbors(state, index):
            force = force + self._repulsion(me - state.positions[j], 2 * self.radius)
        for centre, obstacle_radius in self.obstacles():
            force = force + self._repulsion(me - centre, obstacle_radius + self.radius)
        return limit(force, self.params.cruise_speed)


# --------------------------------------------------------------------------
# social forces
# --------------------------------------------------------------------------


@register_behavior("social_force")
class SocialForce(NavigationBehavior):
    """The social force model (Helbing and Molnár 1995).

    A pedestrian model, and the one most later crowd simulators descend
    from. Each agent accelerates toward its preferred velocity with a
    relaxation time ``tau`` -- the term that makes it a second-order model,
    with momentum -- and is pushed away from others and from obstacles by a
    force that decays exponentially with the gap between their surfaces:
    ``A exp((r_ij - d_ij) / B)``. ``A`` is the strength, ``B`` the range.

    The anisotropy factor of Helbing, Farkas and Vicsek (2000) is included:
    agents react more to what is in front of them than behind, weighted by
    ``lambda_`` (1 makes it isotropic).

    Like every force model it can be tuned into oscillation or pushed through
    contact; the tests report what the default gains do rather than promise.
    """

    name = "social_force"
    output = "acceleration"

    def __init__(
        self,
        tau: float = 0.5,
        strength: float = 6.0,
        range_: float = 0.6,
        lambda_: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tau = tau
        self.strength = strength
        self.range_ = range_
        self.lambda_ = lambda_

    def _push(
        self, me: np.ndarray, other: np.ndarray, gap_radius: float, heading: np.ndarray
    ):
        offset = me - other
        distance = float(np.linalg.norm(offset))
        if distance < 1e-9:
            return np.zeros_like(offset)
        direction = offset / distance
        magnitude = self.strength * math.exp((gap_radius - distance) / self.range_)
        # Anisotropy: cos of the angle between my heading and the direction *to* the other.
        speed = float(np.linalg.norm(heading))
        if speed > 1e-9:
            cos = float(np.dot(heading / speed, -direction))
            magnitude *= self.lambda_ + (1 - self.lambda_) * 0.5 * (1 + cos)
        return magnitude * direction

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        me = state.positions[index]
        velocity = state.velocities[index]
        driving = (self.preferred_velocity(state, index) - velocity) / self.tau
        force = driving
        for j in self.neighbors(state, index):
            force = force + self._push(
                me, state.positions[j], 2 * self.radius, velocity
            )
        for centre, obstacle_radius in self.obstacles():
            force = force + self._push(
                me, centre, self.radius + obstacle_radius, velocity
            )
        return limit(force, self.params.max_acceleration)


# --------------------------------------------------------------------------
# a standard test problem
# --------------------------------------------------------------------------


def circle_swap(
    n: int,
    radius: float = 10.0,
    dimension: int = 2,
    jitter: float = 0.1,
    seed: int = 0,
) -> Tuple[SwarmState, np.ndarray]:
    """Agents on a circle (or sphere), each bound for the antipodal point.

    The benchmark every reciprocal-avoidance paper shows, because every path
    crosses the centre at once and a method that cannot negotiate a crowd
    fails there visibly. Returns ``(state, goals)`` for
    :class:`~pymapf.swarm.simulator.SwarmSimulator` and any
    :class:`NavigationBehavior`.

    ``jitter`` perturbs the start positions. It is on by default because the
    *exactly* symmetric problem deadlocks every symmetric control law: all
    agents reach the centre at the same instant, every avoidance term cancels
    by symmetry, and the swarm freezes. The reference implementations break
    the tie with noise too (RVO2's circle demo perturbs the preferred
    velocity); set ``jitter=0`` to see the deadlock, which one test does.
    """
    rng = np.random.default_rng(seed)
    if dimension == 2:
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        positions = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    elif dimension == 3:
        # Fibonacci sphere: near-uniform, deterministic.
        k = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * k / n)
        theta = math.pi * (1 + 5**0.5) * k
        positions = (
            np.stack(
                [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
                axis=1,
            )
            * radius
        )
    else:
        raise ValueError("circle_swap is defined for 2 or 3 dimensions")
    if jitter:
        positions = positions + rng.normal(0, jitter, size=positions.shape)
    goals = -positions
    return SwarmState(positions, np.zeros_like(positions)), goals
