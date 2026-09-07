"""Piecewise polynomial trajectories in continuous space.

A trajectory here is not a sequence of cells or of waypoints joined by
straight lines: it is a vector-valued polynomial of time, piecewise over a
few segments, differentiable as many times as the objective needs. That is
what "in real space" costs and buys -- there is no grid, positions are
floats, and velocity, acceleration, jerk and snap are exact derivatives
rather than finite differences of a sampled path.

Each segment is written in *normalised* time ``s = (t - t_k) / T_k`` with
``s`` in ``[0, 1]``, so a coefficient's magnitude does not depend on how
long the segment lasts. The derivative with respect to real time carries the
factor ``T_k**-d``, which :func:`derivative_row` applies. This matters: in
raw time the monomials ``t**7`` over a ten-second segment span seven orders
of magnitude and the least-squares systems in :mod:`pymapf.trajectory.qp`
lose most of their digits to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["PiecewisePolynomial", "cost_matrix", "derivative_row"]


def _falling_factorial(j: int, d: int) -> float:
    """``j! / (j - d)!`` -- the coefficient a ``d``-th derivative pulls down."""
    if d > j:
        return 0.0
    return float(math.factorial(j) // math.factorial(j - d))


def derivative_row(order: int, duration: float, s: float, d: int = 0) -> np.ndarray:
    """The row ``r`` with ``r @ c`` equal to the ``d``-th time derivative.

    ``c`` are one segment's coefficients in normalised time and ``s`` is the
    normalised time within the segment; the returned row already includes
    the ``duration ** -d`` chain-rule factor, so ``r @ c`` is in units of
    distance per second to the ``d``.
    """
    row = np.zeros(order + 1)
    if d > order:
        return row
    scale = duration**-d if d else 1.0
    for j in range(d, order + 1):
        row[j] = _falling_factorial(j, d) * (s ** (j - d)) * scale
    return row


def cost_matrix(order: int, duration: float, d: int = 4) -> np.ndarray:
    """``Q`` with ``c @ Q @ c`` equal to the integral of the squared ``d``-th
    derivative over one segment.

    Exact, not quadrature: the integrand is a polynomial, so
    ``∫₀ᵀ (p⁽ᵈ⁾)² dt`` is a finite sum. In normalised time the integral
    carries ``T**(1 - 2d)``, which is why a longer segment is cheaper --
    flying the same shape slower costs less snap, and the optimiser would
    stretch time forever if the duration were free. It is not: the caller
    fixes the time allocation.
    """
    q = np.zeros((order + 1, order + 1))
    if d > order:
        return q
    for i in range(d, order + 1):
        for j in range(d, order + 1):
            power = i + j - 2 * d + 1
            q[i, j] = (
                _falling_factorial(i, d)
                * _falling_factorial(j, d)
                / power
                * duration ** (1 - 2 * d)
            )
    return 0.5 * (q + q.T)  # symmetric by construction; enforce it exactly


@dataclass(frozen=True)
class PiecewisePolynomial:
    """One vehicle's trajectory: ``segments`` polynomials joined at knots.

    Args:
        durations: ``T_k`` seconds per segment; the knots are their prefix
            sums and the trajectory spans ``[0, sum(durations)]``.
        coefficients: ``(segments, dimension, order + 1)``, in normalised
            time per segment.

    Evaluating outside ``[0, duration]`` clamps to the endpoints for
    positions and returns zero for derivatives -- before the start and after
    the finish the vehicle is parked, which is what the boundary conditions
    ask for and what makes a fleet's trajectories comparable at any time.
    """

    durations: Tuple[float, ...]
    coefficients: np.ndarray

    def __post_init__(self):
        coefficients = np.asarray(self.coefficients, dtype=float)
        if coefficients.ndim != 3:
            raise ValueError("coefficients must be (segments, dimension, order + 1)")
        if len(self.durations) != coefficients.shape[0]:
            raise ValueError(
                "%d durations for %d segments"
                % (len(self.durations), coefficients.shape[0])
            )
        if any(t <= 0 for t in self.durations):
            raise ValueError("every segment must last a positive time")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "durations", tuple(float(t) for t in self.durations))
        object.__setattr__(self, "_knots", np.cumsum((0.0,) + self.durations))

    # -- shape ----------------------------------------------------------
    @property
    def segments(self) -> int:
        return self.coefficients.shape[0]

    @property
    def dimension(self) -> int:
        return self.coefficients.shape[1]

    @property
    def order(self) -> int:
        return self.coefficients.shape[2] - 1

    @property
    def duration(self) -> float:
        return float(self._knots[-1])

    @property
    def knots(self) -> np.ndarray:
        """Segment boundaries in seconds, including 0 and the end."""
        return self._knots.copy()

    # -- evaluation -----------------------------------------------------
    def _locate(self, t: float) -> Tuple[int, float]:
        """``(segment, normalised time)`` for a time in seconds."""
        if t <= 0.0:
            return 0, 0.0
        if t >= self.duration:
            return self.segments - 1, 1.0
        k = int(np.searchsorted(self._knots, t, side="right") - 1)
        k = min(max(k, 0), self.segments - 1)
        return k, (t - self._knots[k]) / self.durations[k]

    def __call__(self, t: float, derivative: int = 0) -> np.ndarray:
        """Position (``derivative=0``) or a derivative of it, at time ``t``."""
        outside = t < 0.0 or t > self.duration
        if outside and derivative > 0:
            return np.zeros(self.dimension)
        k, s = self._locate(t)
        row = derivative_row(self.order, self.durations[k], s, derivative)
        return self.coefficients[k] @ row

    def position(self, t: float) -> np.ndarray:
        return self(t, 0)

    def velocity(self, t: float) -> np.ndarray:
        return self(t, 1)

    def acceleration(self, t: float) -> np.ndarray:
        return self(t, 2)

    def sample(self, times: Sequence[float], derivative: int = 0) -> np.ndarray:
        """``(len(times), dimension)`` of values at ``times``."""
        return np.array([self(float(t), derivative) for t in times])

    def speed(self, times: Sequence[float]) -> np.ndarray:
        return np.linalg.norm(self.sample(times, 1), axis=1)

    def path_length(self, samples: int = 400) -> float:
        points = self.sample(np.linspace(0.0, self.duration, samples))
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    # -- transformation -------------------------------------------------
    def dilated(self, factor: float) -> "PiecewisePolynomial":
        """The same *shape*, flown ``factor`` times slower.

        Every segment's duration is multiplied and the normalised
        coefficients are untouched, so the position at normalised time is
        unchanged while speed scales by ``1 / factor`` and acceleration by
        ``1 / factor**2``. Dilating a whole fleet by one factor therefore
        preserves the geometry exactly -- every pair is as far apart at
        corresponding times as it was -- which is what makes it a safe way
        to meet a speed or acceleration limit after the fact.
        """
        if factor <= 0:
            raise ValueError("the dilation factor must be positive")
        return PiecewisePolynomial(
            tuple(t * factor for t in self.durations), self.coefficients.copy()
        )

    def __repr__(self) -> str:
        return "PiecewisePolynomial(segments=%d, dim=%d, order=%d, duration=%.2fs)" % (
            self.segments,
            self.dimension,
            self.order,
            self.duration,
        )


def _time_allocation(
    points: np.ndarray,
    duration: Optional[float],
    speed: float,
    floor_fraction: float = 0.3,
) -> List[float]:
    """Segment durations, roughly proportional to the distance each covers.

    A leg twice as long gets about twice the time, which is what the
    trapezoidal intuition says and a far better starting point than an even
    split. But strict proportionality is a trap when the waypoints come from
    a grid path: a one-cell leg next to a ten-cell one would get a
    twentieth of the flight, and the snap cost of a segment scales as
    ``T**-7``, so that single short segment would dominate the objective and
    the fleet would fly the whole mission badly to flatten it. Each segment
    therefore gets at least ``floor_fraction`` of the even share, and the
    remainder is distributed by length.
    """
    legs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    legs = np.maximum(legs, 1e-9)
    total = duration if duration is not None else float(legs.sum() / speed)
    if total <= 0:
        raise ValueError("duration must be positive")
    n = len(legs)
    floor = floor_fraction * total / n
    free = total - n * floor
    if free <= 0:  # pragma: no cover - only if floor_fraction >= 1
        return [total / n] * n
    return list(floor + free * legs / legs.sum())
