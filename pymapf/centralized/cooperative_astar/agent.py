"""
An agent is defined by an Id, a start position and a goal position. 
To do: add radius
"""

import logging

class Agent:
    def __init__(
        self, ident, init_pos, goal_pos, allow_diagonals=False, nodes_dict=None
    ):
        self.ident = ident
        self.init_pos = init_pos
        self.goal_pos = goal_pos
        self.nodes_dict = nodes_dict
        self.plan = []
        self.conflicts_found = 0
        self.opened_nodes = 0
        self.allow_diagonals = allow_diagonals

    def in_conflict(self, current_state, future_state, other_agents_paths):
        for key, val in other_agents_paths.items():
            if key == self.ident:
                continue

            try:
                if future_state.x == val[-1].x and future_state.y == val[-1].y:
                    logging.debug("Found conflict between agent %s and agent %s" % (self.ident, key))
                    return True
            except BaseException as e:
                logging.debug("Agent %s path is empty: %s" % (key, str(e)))

            if self.allow_diagonals:
                conflicts = [
                    future_state,
                    current_state,
                ]
            else:
                conflicts = [
                    future_state,
                    current_state,
                ]

            for c in conflicts:
                if c in val:
                    logging.warning("Found conflict between agent %s and agent %s" % (self.ident, key))
                    self.conflicts_found += 1
                    return True
        return False

    def __str__(self):
        return "Id: %d | Init: [%d;%d] | Goal: [%d;%d]" % (
            self.ident,
            self.init_pos[0],
            self.init_pos[1],
            self.goal_pos[0],
            self.goal_pos[1],
        )
