"""Time-parameterised motion along a discrete path.

A MAPF solver answers *where* each agent is at integer timesteps. A robot needs
*when*, in seconds, with a speed it can reach and an acceleration it can
survive. This module is the second half: given the distance of one move and the
agent's limits, :class:`MotionProfile` says how long the move takes and where
the agent is partway through it; :class:`Trajectory` strings those moves and the
dwells between them into a continuous function of time; :class:`TrajectorySet`
holds one per agent and answers the only question that matters for a fleet --
how close did any two of them get.

Profiles are rest-to-rest: an agent is stationary at every waypoint. That is the
conservative choice (a real controller would carry speed through a straight
run) and it is what makes the timing of one move independent of the next, so
the scheduler in :mod:`~pymapf.kinodynamic.schedule` can treat each move as a
fixed duration. The cost is a slower trajectory, never an unsafe one.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

Point = Tuple[float, ...]

__all__ = [
    "MotionProfile",
    "Segment",
    "Stay",
    "Trajectory",
    "TrajectorySet",
    "distance",
    "lerp",
]


def distance(a: Point, b: Point) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def lerp(a: Point, b: Point, fraction: float) -> Point:
    return tuple(x + (y - x) * fraction for x, y in zip(a, b))


# --------------------------------------------------------------------------
# one move
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionProfile:
    """How far along a move of length ``length`` an agent is after ``t`` seconds.

    With only a speed limit the profile is a straight line at ``v_max``. With an
    acceleration limit it is the classic trapezoid -- accelerate at ``a_max``,
    cruise at ``v_max``, decelerate at ``a_max`` -- or, when the move is too
    short to reach cruise speed, the triangle that replaces it. Both start and
    end at rest.

    Attributes:
        length: distance of the move.
        v_max: speed limit (> 0).
        a_max: acceleration limit, or ``None`` for a purely kinematic profile.
    """

    length: float
    v_max: float
    a_max: Optional[float] = None

    def __post_init__(self):
        if self.v_max <= 0:
            raise ValueError("v_max must be positive, got %r" % (self.v_max,))
        if self.a_max is not None and self.a_max <= 0:
            raise ValueError("a_max must be positive or None, got %r" % (self.a_max,))
        if self.length < 0:
            raise ValueError("length must be non-negative, got %r" % (self.length,))

    # -- derived timings ------------------------------------------------
    @property
    def peak_speed(self) -> float:
        """The speed actually reached: ``v_max`` unless the move is too short."""
        if self.a_max is None:
            return self.v_max
        # Distance needed to accelerate to v_max and back down again.
        if self.length >= self.v_max**2 / self.a_max:
            return self.v_max
        return math.sqrt(self.a_max * self.length)

    @property
    def ramp_time(self) -> float:
        """Seconds spent accelerating (and, symmetrically, decelerating)."""
        if self.a_max is None:
            return 0.0
        return self.peak_speed / self.a_max

    @property
    def duration(self) -> float:
        if self.length == 0:
            return 0.0
        if self.a_max is None:
            return self.length / self.v_max
        v = self.peak_speed
        cruise = (self.length - v**2 / self.a_max) / v
        return 2 * self.ramp_time + max(cruise, 0.0)

    # -- evaluation -----------------------------------------------------
    def travelled(self, t: float) -> float:
        """Distance covered ``t`` seconds after the move starts (clamped)."""
        if t <= 0 or self.length == 0:
            return 0.0
        duration = self.duration
        if t >= duration:
            return self.length
        if self.a_max is None:
            return self.v_max * t
        a, ramp, v = self.a_max, self.ramp_time, self.peak_speed
        if t < ramp:
            return 0.5 * a * t * t
        if t > duration - ramp:
            remaining = duration - t
            return self.length - 0.5 * a * remaining * remaining
        return 0.5 * a * ramp * ramp + v * (t - ramp)

    def time_to_travel(self, s: float) -> float:
        """Seconds until ``s`` of the move has been covered -- the inverse of
        :meth:`travelled`. Clamped to ``[0, duration]``.

        This is how a *spatial* safety margin becomes a *temporal* one: the
        time the departing agent needs to open that gap, under its own
        profile. Dividing by ``v_max`` would assume it leaves at full speed,
        which an agent accelerating from rest does not.
        """
        if s <= 0 or self.length == 0:
            return 0.0
        if s >= self.length:
            return self.duration
        if self.a_max is None:
            return s / self.v_max
        a, ramp, v = self.a_max, self.ramp_time, self.peak_speed
        ramp_distance = 0.5 * a * ramp * ramp
        if s <= ramp_distance:
            return math.sqrt(2 * s / a)
        if s >= self.length - ramp_distance:
            return self.duration - math.sqrt(2 * (self.length - s) / a)
        return ramp + (s - ramp_distance) / v

    def speed(self, t: float) -> float:
        """Instantaneous speed ``t`` seconds into the move."""
        if t < 0 or t > self.duration or self.length == 0:
            return 0.0
        if self.a_max is None:
            return self.v_max
        a, ramp, v = self.a_max, self.ramp_time, self.peak_speed
        if t < ramp:
            return a * t
        if t > self.duration - ramp:
            return a * (self.duration - t)
        return v


# --------------------------------------------------------------------------
# one agent
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Stay:
    """An agent parked at a vertex from ``arrive`` until ``depart``.

    ``depart`` is ``inf`` for the final stay: an agent that has reached its goal
    stays there, exactly as :meth:`pymapf.Solution.position_at` assumes.
    """

    vertex: object
    position: Point
    arrive: float
    depart: float

    @property
    def dwell(self) -> float:
        return self.depart - self.arrive


@dataclass(frozen=True)
class Segment:
    """One move, from the moment the agent leaves a vertex to the moment it
    arrives at the next."""

    start: float
    end: float
    origin: Point
    destination: Point
    profile: MotionProfile

    def position_at(self, t: float) -> Point:
        if self.profile.length == 0:
            return self.origin
        fraction = self.profile.travelled(t - self.start) / self.profile.length
        return lerp(self.origin, self.destination, fraction)

    def velocity_at(self, t: float) -> Point:
        if self.profile.length == 0:
            return tuple(0.0 for _ in self.origin)
        speed = self.profile.speed(t - self.start)
        return tuple(
            (b - a) / self.profile.length * speed
            for a, b in zip(self.origin, self.destination)
        )


@dataclass
class Trajectory:
    """A single agent's continuous-time motion: stays joined by segments.

    Built by :func:`~pymapf.kinodynamic.schedule.plan_trajectories`; the
    pieces alternate ``stay, segment, stay, ..., stay`` and cover all of
    ``[0, inf)``.
    """

    agent: str
    stays: List[Stay]
    segments: List[Segment] = field(default_factory=list)

    def __post_init__(self):
        if not self.stays:
            raise ValueError("a trajectory needs at least one stay")
        if len(self.segments) != len(self.stays) - 1:
            raise ValueError(
                "%d stays need %d segments, got %d"
                % (len(self.stays), len(self.stays) - 1, len(self.segments))
            )
        self._segment_starts = [segment.start for segment in self.segments]

    @property
    def start(self) -> Point:
        return self.stays[0].position

    @property
    def goal(self) -> Point:
        return self.stays[-1].position

    @property
    def arrival_time(self) -> float:
        """When the agent settles on its goal for good."""
        return self.stays[-1].arrive

    @property
    def path_length(self) -> float:
        return sum(segment.profile.length for segment in self.segments)

    def _piece(self, t: float):
        """The stay or segment active at time ``t``."""
        if t <= self.stays[0].depart or not self.segments:
            return self.stays[0]
        # Last segment that has started by time t.
        index = bisect.bisect_right(self._segment_starts, t) - 1
        segment = self.segments[index]
        if t <= segment.end:
            return segment
        return self.stays[index + 1]

    def position_at(self, t: float) -> Point:
        piece = self._piece(t)
        if isinstance(piece, Segment):
            return piece.position_at(t)
        return piece.position

    def velocity_at(self, t: float) -> Point:
        piece = self._piece(t)
        if isinstance(piece, Segment):
            return piece.velocity_at(t)
        return tuple(0.0 for _ in piece.position)

    def vertices(self) -> List[object]:
        """The discrete path this trajectory realises, one entry per stay."""
        return [stay.vertex for stay in self.stays]

    def __repr__(self) -> str:
        return "Trajectory(%r, stays=%d, arrives=%.2fs, length=%.2f)" % (
            self.agent,
            len(self.stays),
            self.arrival_time,
            self.path_length,
        )


# --------------------------------------------------------------------------
# the fleet
# --------------------------------------------------------------------------


class TrajectorySet:
    """One :class:`Trajectory` per agent, with the fleet-level questions.

    Behaves like a read-only mapping from agent name to trajectory.
    """

    def __init__(self, trajectories: Dict[str, Trajectory]):
        if not trajectories:
            raise ValueError("a TrajectorySet needs at least one trajectory")
        self._trajectories = dict(trajectories)

    # -- mapping protocol ------------------------------------------------
    def __getitem__(self, agent: str) -> Trajectory:
        return self._trajectories[agent]

    def __iter__(self) -> Iterator[str]:
        return iter(self._trajectories)

    def __len__(self) -> int:
        return len(self._trajectories)

    def __contains__(self, agent) -> bool:
        return agent in self._trajectories

    def items(self):
        return self._trajectories.items()

    def values(self):
        return self._trajectories.values()

    @property
    def agents(self) -> List[str]:
        return list(self._trajectories)

    # -- fleet metrics ---------------------------------------------------
    @property
    def makespan(self) -> float:
        """Seconds until the last agent settles on its goal."""
        return max(t.arrival_time for t in self._trajectories.values())

    @property
    def sum_of_arrival_times(self) -> float:
        """The continuous analogue of sum-of-costs."""
        return sum(t.arrival_time for t in self._trajectories.values())

    def positions_at(self, t: float) -> Dict[str, Point]:
        return {name: traj.position_at(t) for name, traj in self._trajectories.items()}

    def sample(self, dt: float = 0.05, until: Optional[float] = None):
        """Positions of every agent on a regular time grid.

        Returns ``(times, positions)`` where ``positions[agent]`` is a list of
        points aligned with ``times``. Pure Python so the core stays free of
        numpy; callers with numpy can ``np.asarray`` the result.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        horizon = self.makespan if until is None else until
        n = int(math.floor(horizon / dt)) + 1
        times = [i * dt for i in range(n)]
        if times[-1] < horizon:
            times.append(horizon)
        positions = {
            name: [traj.position_at(t) for t in times]
            for name, traj in self._trajectories.items()
        }
        return times, positions

    def min_separation(self, dt: float = 0.05) -> Tuple[float, float, Tuple[str, str]]:
        """The closest any two agents come, sampled every ``dt`` seconds.

        Returns ``(distance, time, (agent_a, agent_b))``. Sampling is the
        honest tool here: the schedule guarantees separation in *time* (see
        :mod:`~pymapf.kinodynamic.schedule`), and this measures what that
        bought in *space* for the actual speeds involved.
        """
        names = self.agents
        if len(names) < 2:
            return math.inf, 0.0, ("", "")
        times, positions = self.sample(dt)
        best = (math.inf, 0.0, (names[0], names[1]))
        for k, t in enumerate(times):
            for i in range(len(names)):
                pi = positions[names[i]][k]
                for j in range(i + 1, len(names)):
                    d = distance(pi, positions[names[j]][k])
                    if d < best[0]:
                        best = (d, t, (names[i], names[j]))
        return best

    def max_speed(self, dt: float = 0.05) -> float:
        """Largest sampled speed of any agent -- should never exceed its limit."""
        times, _ = self.sample(dt)
        return max(
            math.sqrt(sum(v * v for v in traj.velocity_at(t)))
            for traj in self._trajectories.values()
            for t in times
        )

    def __repr__(self) -> str:
        return "TrajectorySet(agents=%d, makespan=%.2fs)" % (len(self), self.makespan)
