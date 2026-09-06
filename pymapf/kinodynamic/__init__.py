"""From discrete plans to trajectories a robot can follow.

    import pymapf
    from pymapf.kinodynamic import KinematicLimits, plan_trajectories

    scenario = pymapf.build_scenario("warehouse", n_agents=8, seed=3)
    solution = pymapf.solve(scenario.to_problem(), "cbs")

    limits = KinematicLimits(v_max=1.5, a_max=2.0, safety_distance=0.5)
    trajectories = plan_trajectories(solution, limits=limits)

    trajectories.makespan                 # seconds, not timesteps
    trajectories["a"].position_at(3.25)   # (row, col) as floats
    trajectories.min_separation()         # the closest any two agents came

The schedule is MAPF-POST (Hönig et al. 2016) with rest-to-rest motion
profiles; see :mod:`pymapf.kinodynamic.schedule` for exactly what it
guarantees and what it does not. Pure standard library, like the solvers.
"""

from .schedule import (
    InfeasibleScheduleError,
    KinematicLimits,
    plan_trajectories,
    required_margin,
    stays_of,
)
from .trajectory import (
    MotionProfile,
    Segment,
    Stay,
    Trajectory,
    TrajectorySet,
    distance,
    lerp,
)

__all__ = [
    "InfeasibleScheduleError",
    "KinematicLimits",
    "MotionProfile",
    "Segment",
    "Stay",
    "Trajectory",
    "TrajectorySet",
    "distance",
    "lerp",
    "plan_trajectories",
    "required_margin",
    "stays_of",
]
