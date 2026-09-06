"""Turn a discrete MAPF plan into a kinematically feasible schedule.

The idea is Hönig et al.'s MAPF-POST (ICAPS 2016): keep the *order* in which
the discrete plan has agents visit each vertex, throw away its *timing*, and
re-derive the timing as the solution of a simple temporal network whose
constraints are physical rather than unit-step. Two kinds of constraint:

* **travel** -- an agent cannot leave a vertex before it has arrived there,
  and a move takes at least as long as its :class:`MotionProfile` says;
* **safety** -- if the discrete plan has agent A at vertex ``v`` before agent
  B, then B may arrive at ``v`` only ``safety_time`` seconds after A *starts to
  leave* it.

Every constraint has the form ``x >= y + w``, so the earliest schedule is the
longest path from a virtual source, found by Bellman-Ford. Using the moment A
*starts* leaving rather than the moment it has fully left is what makes
rotations feasible: three agents cycling around a loop all start at once, and
the spacing between them is the edge length. It also means the safety margin
cannot exceed a move's duration -- ask for more and the network has a positive
cycle, which :class:`InfeasibleScheduleError` reports as exactly that.

The paper's safety margin is one number. Here it is derived per hand-over by
:func:`required_margin`, which simulates A leaving and B arriving under their
actual profiles and the actual angle between the two moves, and finds the
smallest delay at which they never come within ``safety_distance``. That
matters because rest-to-rest profiles are slow at both ends: A creeps away
from the vertex while B creeps onto it, and a margin sized for full speed
lets a right-angle hand-over get far closer than asked. The tests check the
sampled minimum over whole plans against the distance requested.

There is one respect in which the margin is *not* a guarantee: it covers each
pair's hand-over at a shared vertex, not two agents passing on adjacent
vertices without ever sharing one. On a unit grid those keep one cell apart,
which is fine for ``safety_distance <= 1``; ask for more than the edge length
and :meth:`TrajectorySet.min_separation` is the check, as it is for anything
else this model does not see.

Deviations from the paper: profiles are rest-to-rest at every vertex (the
paper allows any speed within limits, at the cost of a second bound per move
and an LP), and there is no upper bound on dwell, so an agent may wait
indefinitely for a slower one. Both are simplifications that keep the network
a plain longest-path problem and the core free of a solver dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from .trajectory import (
    MotionProfile,
    Point,
    Segment,
    Stay,
    Trajectory,
    TrajectorySet,
    distance,
)

__all__ = [
    "KinematicLimits",
    "InfeasibleScheduleError",
    "plan_trajectories",
    "required_margin",
    "stays_of",
]


@dataclass(frozen=True)
class KinematicLimits:
    """What one agent can do.

    Args:
        v_max: top speed, in distance units per second.
        a_max: acceleration limit; ``None`` means the agent changes speed
            instantly, which is the classical MAPF-POST assumption.
        safety_distance: the closest another agent may come during a
            hand-over at a shared vertex. Converted to a per-hand-over delay
            by :func:`required_margin`, which accounts for both agents'
            profiles and the angle between their moves.
        safety_time: seconds a vertex must have been left before the next
            agent arrives. A fixed rule, the paper's original; overrides
            ``safety_distance`` when given.
    """

    v_max: float = 1.0
    a_max: Optional[float] = None
    safety_distance: float = 0.0
    safety_time: Optional[float] = None

    def __post_init__(self):
        if self.v_max <= 0:
            raise ValueError("v_max must be positive")
        if self.a_max is not None and self.a_max <= 0:
            raise ValueError("a_max must be positive or None")
        if self.safety_distance < 0:
            raise ValueError("safety_distance cannot be negative")
        if self.safety_time is not None and self.safety_time < 0:
            raise ValueError("safety_time cannot be negative")

    def profile(self, length: float) -> MotionProfile:
        return MotionProfile(length, self.v_max, self.a_max)


class InfeasibleScheduleError(ValueError):
    """No timing satisfies the constraints.

    For a valid discrete plan this only happens when the safety margin is
    longer than some move's duration on a cycle of agents that rotate
    simultaneously: everyone must wait for everyone else, forever.
    """


# --------------------------------------------------------------------------
# discrete path -> stays
# --------------------------------------------------------------------------


def stays_of(path: Sequence[Hashable]) -> List[Tuple[Hashable, int, int]]:
    """Collapse a discrete path into ``(vertex, first_t, last_t)`` stays.

    A wait action in the discrete plan is the same vertex at consecutive
    timesteps; here it is one stay with a longer dwell, since the continuous
    schedule decides dwell for itself.
    """
    if not path:
        raise ValueError("empty path")
    stays: List[Tuple[Hashable, int, int]] = []
    first = 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[first]:
            stays.append((path[first], first, t - 1))
            first = t
    return stays


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------


def required_margin(
    vertex: Point,
    leaving_to: Point,
    leaving: MotionProfile,
    arriving_from: Point,
    arriving: MotionProfile,
    safety_distance: float,
    samples: int = 48,
) -> float:
    """How long after A starts leaving ``vertex`` may B arrive there, so that
    the two are never closer than ``safety_distance``.

    A scalar rule cannot answer this. A leaves from rest, so at first it is
    barely moving; B arrives *to* rest, so it spends the end of its approach
    almost on the vertex. Whether that is dangerous depends on the angle
    between the two moves -- straight following is benign, a right-angle
    hand-over is the worst case on a grid -- and on both profiles. So the
    interaction is simulated: for a candidate delay, sample the distance over
    B's whole approach with A moving as scheduled, and search for the smallest
    delay whose minimum clears ``safety_distance``.

    The search is a coarse scan refined by bisection, which does not assume
    the separation is monotone in the delay (on an acute angle it need not
    be) -- only that the scan is fine enough to bracket the first crossing.
    Returns ``0`` when no delay is needed and at most ``leaving.duration +
    arriving.duration``, by which time A has completed its move and B has
    not started; a ``safety_distance`` longer than the edges can still be
    violated then, and is outside what one hand-over can promise.
    """
    if safety_distance <= 0:
        return 0.0
    to_next = tuple(w - c for w, c in zip(leaving_to, vertex))
    from_prev = tuple(c - u for c, u in zip(vertex, arriving_from))
    len_a, len_b = leaving.length, arriving.length
    dur_a, dur_b = leaving.duration, arriving.duration

    def a_at(t: float) -> Point:
        if t <= 0 or len_a == 0:
            return vertex
        f = leaving.travelled(t) / len_a
        return tuple(c + d * f for c, d in zip(vertex, to_next))

    def b_at(t: float, arrival: float) -> Point:
        if len_b == 0:
            return vertex
        f = arriving.travelled(t - (arrival - dur_b)) / len_b
        return tuple(u + d * f for u, d in zip(arriving_from, from_prev))

    def separation(arrival: float) -> float:
        start = min(0.0, arrival - dur_b)
        span = arrival - start
        best = math.inf
        for k in range(samples + 1):
            t = start + span * k / samples
            best = min(best, distance(a_at(t), b_at(t, arrival)))
        return best

    cap = dur_a + dur_b
    if separation(0.0) >= safety_distance:
        return 0.0
    # Coarse scan to bracket the first delay that works.
    steps = 32
    previous = 0.0
    for k in range(1, steps + 1):
        candidate = cap * k / steps
        if separation(candidate) >= safety_distance:
            lo, hi = previous, candidate
            break
        previous = candidate
    else:
        return cap
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if separation(mid) >= safety_distance:
            hi = mid
        else:
            lo = mid
    return hi


def _resolve_positions(paths, graph, positions, cell_size) -> Dict[Hashable, Point]:
    """Coordinates for every vertex that appears in any path."""
    vertices = {v for path in paths.values() for v in path}
    if positions is not None:
        table = {v: tuple(float(x) for x in positions[v]) for v in vertices}
    elif graph is not None and getattr(graph, "positions", None):
        table = {v: tuple(float(x) for x in graph.positions[v]) for v in vertices}
    else:
        # Grid cells are their own coordinates.
        try:
            table = {v: tuple(float(x) * cell_size for x in v) for v in vertices}
        except (TypeError, ValueError):
            raise ValueError(
                "vertices are not coordinate tuples; pass positions= or a graph "
                "with a .positions layout"
            )
    return table


# --------------------------------------------------------------------------
# the temporal network
# --------------------------------------------------------------------------


def plan_trajectories(
    solution,
    graph=None,
    positions: Optional[Mapping[Hashable, Sequence[float]]] = None,
    limits: Optional[KinematicLimits] = None,
    limits_by_agent: Optional[Mapping[str, KinematicLimits]] = None,
    cell_size: float = 1.0,
) -> TrajectorySet:
    """Schedule a discrete :class:`pymapf.Solution` under kinematic limits.

    Args:
        solution: a *valid* discrete plan. Validity is what makes the visit
            order at each vertex well-defined; an invalid plan is rejected.
        graph: the map the plan was made on, used for vertex coordinates when
            it carries a ``positions`` layout (an ``ExplicitGraph``).
        positions: explicit ``{vertex: (x, y, ...)}`` coordinates, taking
            precedence over ``graph``. Grid cells need neither.
        limits: limits shared by every agent (default: ``v_max = 1``).
        limits_by_agent: per-agent overrides, for heterogeneous fleets.
        cell_size: metres per grid cell when cells are used as coordinates.

    Returns:
        A :class:`TrajectorySet`, earliest-possible under the constraints.

    Raises:
        InfeasibleScheduleError: see the class docstring.
        ValueError: on an invalid plan or unresolvable coordinates.
    """
    if not solution.is_valid():
        conflict = solution.first_conflict()
        raise ValueError(
            "the discrete plan has a %s conflict between %r and %r at t=%d; "
            "schedule only valid plans"
            % (conflict.kind, conflict.a, conflict.b, conflict.t)
        )
    paths = solution.paths
    if not paths:
        raise ValueError("the solution has no agents")

    shared = limits or KinematicLimits()
    overrides = dict(limits_by_agent or {})

    def limits_for(agent: str) -> KinematicLimits:
        return overrides.get(agent, shared)

    coords = _resolve_positions(paths, graph, positions, cell_size)

    # -- variables: one per departure --------------------------------------
    # dep[(agent, k)] is when the agent leaves its k-th stay. Arrival at stay
    # k+1 is dep[(agent, k)] + duration of that move -- not a variable, which
    # is what keeps every constraint a plain difference.
    stays: Dict[str, List[Tuple[Hashable, int, int]]] = {
        agent: stays_of(path) for agent, path in paths.items()
    }
    durations: Dict[Tuple[str, int], float] = {}
    profiles: Dict[Tuple[str, int], MotionProfile] = {}
    for agent, agent_stays in stays.items():
        lim = limits_for(agent)
        for k in range(len(agent_stays) - 1):
            a, b = agent_stays[k][0], agent_stays[k + 1][0]
            profile = lim.profile(distance(coords[a], coords[b]))
            profiles[(agent, k)] = profile
            durations[(agent, k)] = profile.duration

    variables = list(durations)  # every (agent, k) with a departure
    index = {var: i for i, var in enumerate(variables)}

    # Edges (src, dst, w) meaning  t[dst] >= t[src] + w.  src=None is the
    # virtual source at time 0.
    edges: List[Tuple[Optional[int], int, float]] = []
    for var in variables:
        edges.append((None, index[var], 0.0))

    # travel + dwell: leave stay k+1 no earlier than arriving there.
    for agent, agent_stays in stays.items():
        for k in range(len(agent_stays) - 2):
            edges.append(
                (index[(agent, k)], index[(agent, k + 1)], durations[(agent, k)])
            )

    # safety: consecutive visitors of a vertex, in discrete-time order.
    visits: Dict[Hashable, List[Tuple[int, str, int]]] = {}
    for agent, agent_stays in stays.items():
        for k, (vertex, first, _last) in enumerate(agent_stays):
            visits.setdefault(vertex, []).append((first, agent, k))

    for vertex, visitors in visits.items():
        visitors.sort()
        for (_, a, ka), (_, b, kb) in zip(visitors, visitors[1:]):
            if a == b:
                # The same agent revisiting a vertex: its own travel chain
                # already orders the two stays.
                continue
            # A must have a departure (it is not parked here for good) and B
            # must have an arrival (it did not start here); a valid plan
            # guarantees both, so these are assertions rather than branches.
            assert (
                ka < len(stays[a]) - 1
            ), "%r parks at %r but %r visits it later; plan is not valid" % (
                a,
                vertex,
                b,
            )
            assert kb > 0, "%r starts at %r but %r was there earlier" % (b, vertex, a)
            # The margin is derived from this particular hand-over: A's
            # departing move, B's arriving move, and the angle between them.
            # A cautious agent following a reckless one still gets its own
            # distance, so the larger of the two requirements applies.
            lim_a, lim_b = limits_for(a), limits_for(b)
            explicit = [
                lim.safety_time for lim in (lim_a, lim_b) if lim.safety_time is not None
            ]
            wanted = max(lim_a.safety_distance, lim_b.safety_distance)
            margin = max(explicit) if explicit else 0.0
            if wanted > 0 and not explicit:
                margin = required_margin(
                    coords[vertex],
                    coords[stays[a][ka + 1][0]],
                    profiles[(a, ka)],
                    coords[stays[b][kb - 1][0]],
                    profiles[(b, kb - 1)],
                    wanted,
                )
            # arrive(B, kb) = dep(B, kb-1) + duration >= dep(A, ka) + margin
            edges.append(
                (
                    index[(a, ka)],
                    index[(b, kb - 1)],
                    margin - durations[(b, kb - 1)],
                )
            )

    times = _longest_path(len(variables), edges)

    # -- assemble ---------------------------------------------------------
    trajectories: Dict[str, Trajectory] = {}
    for agent, agent_stays in stays.items():
        stay_objects: List[Stay] = []
        segments: List[Segment] = []
        arrive = 0.0
        for k, (vertex, _first, _last) in enumerate(agent_stays):
            last = k == len(agent_stays) - 1
            depart = math.inf if last else times[index[(agent, k)]]
            stay_objects.append(Stay(vertex, coords[vertex], arrive, depart))
            if not last:
                profile = profiles[(agent, k)]
                next_vertex = agent_stays[k + 1][0]
                end = depart + profile.duration
                segments.append(
                    Segment(depart, end, coords[vertex], coords[next_vertex], profile)
                )
                arrive = end
        trajectories[agent] = Trajectory(agent, stay_objects, segments)
    return TrajectorySet(trajectories)


def _longest_path(n: int, edges) -> List[float]:
    """Earliest times satisfying every ``t[dst] >= t[src] + w``.

    Bellman-Ford, maximising. Converges in at most ``n`` passes; a change on
    the pass after that is a positive cycle, i.e. no finite schedule.
    """
    times = [0.0] * n
    for _ in range(n + 1):
        changed = False
        for src, dst, w in edges:
            base = 0.0 if src is None else times[src]
            if times[dst] < base + w - 1e-12:
                times[dst] = base + w
                changed = True
        if not changed:
            return times
    raise InfeasibleScheduleError(
        "the safety margin exceeds a move's duration on a cycle of agents "
        "that must rotate simultaneously; reduce safety_time/safety_distance "
        "or raise v_max"
    )
