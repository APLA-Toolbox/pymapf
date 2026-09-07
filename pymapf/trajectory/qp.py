"""A dense quadratic program solver, enough for one convex subproblem.

    minimise   ½ xᵀ H x + gᵀ x
    subject to A x  = b
               C x <= d

Every subproblem the joint trajectory optimiser solves has this shape: the
cost is the integral of squared snap (quadratic in the coefficients), the
boundary and continuity conditions are linear equalities, and the
*linearised* separation, speed and trust-region constraints are linear
inequalities. So one solver serves the whole layer, and the layer keeps its
only dependency on numpy.

The method is an augmented Lagrangian on the inequalities with a
semismooth-Newton inner solve:

* Inequalities enter the objective as ``(ρ/2)‖(Cx - d + u)₊‖²``, whose
  gradient is piecewise linear -- so for a *fixed* set of violated rows the
  minimiser under ``Ax = b`` is one KKT linear system. Iterating "guess the
  active set, solve, re-derive the active set" is Newton's method on the
  piecewise-linear gradient and converges in a handful of steps.
* The multipliers ``u`` are then updated, and ``ρ`` grows when the
  violation does not fall. This is the standard method of multipliers
  (Hestenes 1969, Powell 1969), which converges without ``ρ`` going to
  infinity -- the reason to prefer it to a plain penalty, whose Hessian
  gets worse the tighter you want the constraints.

An active-set method would give an exact solution in finitely many steps.
This does not; it gives a solution to a tolerance, which is all a
sequential-convex outer loop can use anyway, and it never has to decide
which constraints to drop -- the part of an active-set implementation that
cycles. The tests check it against SLSQP on random dense problems.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

__all__ = ["solve_qp", "QPResult"]


class QPResult(NamedTuple):
    """What the solver found: unpacks as ``(x, converged)`` first."""

    x: np.ndarray
    converged: bool
    violation: float
    steps: int


def _kkt_solve(
    hessian: np.ndarray,
    gradient: np.ndarray,
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
    regularisation: float,
) -> np.ndarray:
    """Minimise ``½xᵀHx + gᵀx`` subject to ``Ax = b`` by one linear solve.

    The KKT matrix is symmetric indefinite. A small multiple of the identity
    on the ``x`` block keeps it non-singular when ``H`` is only positive
    *semi*-definite, which it is here: the snap cost does not see the
    constant and linear coefficients of a segment at all, so ``H`` has a
    null space of exactly the trajectories that cost nothing.
    """
    n = hessian.shape[0]
    h = hessian + regularisation * np.eye(n)
    if a is None or a.size == 0:
        return np.linalg.solve(h, -gradient)
    m = a.shape[0]
    kkt = np.zeros((n + m, n + m))
    kkt[:n, :n] = h
    kkt[:n, n:] = a.T
    kkt[n:, :n] = a
    kkt[n:, n:] = -regularisation * np.eye(m)
    rhs = np.concatenate([-gradient, b])
    try:
        solution = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return solution[:n]


def _violation(
    c: Optional[np.ndarray],
    d: Optional[np.ndarray],
    x: np.ndarray,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
) -> float:
    """The worst constraint residual: inequalities *and* equalities.

    The equalities are enforced by the KKT system rather than by the
    penalty, so they are usually satisfied to machine precision -- but not
    when they are inconsistent, and a solver that reports success on an
    infeasible problem is worse than one that fails.
    """
    worst = 0.0
    if c is not None and c.size:
        worst = max(worst, float(np.maximum(c @ x - d, 0.0).max(initial=0.0)))
    if a is not None and a.size:
        worst = max(worst, float(np.abs(a @ x - b).max(initial=0.0)))
    return worst


def solve_qp(
    hessian: np.ndarray,
    gradient: np.ndarray,
    equality: Optional[np.ndarray] = None,
    equality_rhs: Optional[np.ndarray] = None,
    inequality: Optional[np.ndarray] = None,
    inequality_rhs: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    rho: float = 10.0,
    outer: int = 60,
    inner: int = 25,
    tolerance: float = 1e-8,
    regularisation: float = 1e-9,
) -> QPResult:
    """Minimise ``½xᵀHx + gᵀx`` under ``Ax = b`` and ``Cx <= d``.

    Args:
        hessian: symmetric ``(n, n)``, positive semi-definite.
        gradient: ``(n,)``.
        equality / equality_rhs: ``A`` and ``b``; omit for none.
        inequality / inequality_rhs: ``C`` and ``d``; omit for none.
        x0: a starting point, used only to seed the first active set.
        rho: initial penalty weight. It grows when the violation stalls.
        outer: multiplier updates.
        inner: semismooth-Newton steps per multiplier update.
        tolerance: on both the constraint violation and the step size.
        regularisation: added to the KKT diagonal; see :func:`_kkt_solve`.

    Returns:
        :class:`QPResult`, which unpacks as ``(x, converged)``.
    """
    n = hessian.shape[0]
    hessian = 0.5 * (hessian + hessian.T)
    a = None if equality is None or equality.size == 0 else np.asarray(equality, float)
    b = None if a is None else np.asarray(equality_rhs, float)
    c = (
        None
        if inequality is None or inequality.size == 0
        else np.asarray(inequality, float)
    )
    d = None if c is None else np.asarray(inequality_rhs, float)

    if c is None:
        x = _kkt_solve(hessian, gradient, a, b, regularisation)
        residual = _violation(None, None, x, a, b)
        return QPResult(x, residual <= 1e-6, residual, 1)

    x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()
    if a is not None:
        # Start on the affine set, so that every damped step below stays on
        # it: both endpoints of the line satisfy Ax = b, hence so does every
        # point between them.
        x = x + np.linalg.lstsq(a, b - a @ x, rcond=None)[0]
    u = np.zeros(c.shape[0])  # scaled multipliers, mu / rho
    previous_violation = np.inf
    steps = 0

    def merit(point: np.ndarray, penalty: float, shift: np.ndarray) -> float:
        """The augmented Lagrangian's value -- convex, piecewise quadratic."""
        excess = np.maximum(c @ point - d + shift, 0.0)
        return float(
            0.5 * point @ hessian @ point
            + gradient @ point
            + 0.5 * penalty * (excess @ excess)
        )

    for _ in range(outer):
        before = x.copy()
        x, taken = _newton(
            x, u, rho, hessian, gradient, a, b, c, d, inner, regularisation, merit
        )
        steps += taken
        violation = _violation(c, d, x, a, b)
        movement = float(np.linalg.norm(x - before))
        # Feasibility alone is not optimality: an over-tight penalty can push
        # the iterate well inside the feasible set. Stop only once the
        # multipliers have settled too, which is what makes this the method
        # of multipliers rather than a penalty method.
        if violation <= tolerance and movement <= 1e-7 * max(1.0, np.linalg.norm(x)):
            return QPResult(x, True, violation, steps)
        u = np.maximum(u + c @ x - d, 0.0)
        # Grow the penalty only while the iterate is still infeasible. Growing
        # it once the violation is already at the tolerance buys nothing and
        # costs conditioning: the Newton systems carry rho * C_S^T C_S, and at
        # rho = 1e12 that swamps the cost Hessian entirely.
        if violation > tolerance and violation > 0.5 * previous_violation:
            rho = min(rho * 5.0, 1e8)
        previous_violation = violation
    final = _violation(c, d, x, a, b)
    return QPResult(x, final <= max(tolerance, 1e-6), final, steps)


def _newton(x, u, rho, hessian, gradient, a, b, c, d, inner, regularisation, merit):
    """Minimise the augmented Lagrangian for fixed multipliers.

    For a fixed set of violated rows the subproblem is a plain KKT solve, and
    iterating on that set is Newton's method on a piecewise-linear gradient.
    Undamped it can cycle between two sets; since the merit function is
    convex the Newton direction is a descent direction, so a backtracking
    line search on it makes this converge globally.
    """
    steps = 0
    for _ in range(inner):
        active = (c @ x - d + u) > 0.0
        ca = c[active]
        h = hessian + rho * (ca.T @ ca)
        g = gradient - rho * (ca.T @ (d[active] - u[active]))
        candidate = _kkt_solve(h, g, a, b, regularisation)
        if not np.all(np.isfinite(candidate)):  # pragma: no cover - defensive
            break
        steps += 1
        direction = candidate - x
        reference = merit(x, rho, u)
        length = 1.0
        while length > 1e-6 and merit(x + length * direction, rho, u) > reference:
            length *= 0.5
        if length <= 1e-6:
            break  # no descent left: this multiplier estimate is done
        x = x + length * direction
        if np.linalg.norm(length * direction) <= 1e-12 * max(
            1.0, float(np.linalg.norm(x))
        ):
            break
    return x, steps
