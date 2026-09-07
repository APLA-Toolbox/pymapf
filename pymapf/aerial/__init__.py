"""Aerial fleets: quadrotors assigned to docking stations and flown there
on kinematically feasible, collision-free trajectories.

::

    from pymapf.aerial import Airspace, hovering_fleet, plan_docking

    airspace = Airspace.build(cells=(12, 10, 6), cell_size=1.5,
                              pads=[(1, 2), (1, 5), (1, 8), (8, 2), (8, 5), (8, 8)])
    fleet = hovering_fleet(airspace, n=6, seed=0)
    plan = plan_docking(fleet, airspace)     # assign, route, schedule

    plan.assignment                          # {"q0": "pad-3", ...}
    plan.makespan                            # seconds until the last one is down
    plan.trajectories["q0"].position_at(4.0) # metres, mid-flight
    plan.min_separation()                    # the closest any two came

See :mod:`pymapf.aerial.docking` for the three optimisations this stacks.
"""

from .docking import (
    Airspace,
    DockingPlan,
    DockingStation,
    Quadrotor,
    assign_stations,
    hovering_fleet,
    plan_docking,
)

__all__ = [
    "Airspace",
    "DockingPlan",
    "DockingStation",
    "Quadrotor",
    "assign_stations",
    "hovering_fleet",
    "plan_docking",
]
