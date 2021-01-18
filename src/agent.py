from .common import DIAGONALS

class Agent:
    def __init__(self, id, init_pos, goal_pos, nodes_dict=None):
        self.id = id
        self.init_pos = init_pos
        self.goal_pos = goal_pos
        self.nodes_dict = nodes_dict
        self.plan = []
        self.conflicts_found = 0
        self.opened_nodes = 0

    def in_conflict(self, state, other_agents_paths):
        for key, val in other_agents_paths.items():
            if key == id:
                continue
            
            if DIAGONALS:
                conflict_1 = state
                conflict_2 = state
                conflict_2.t += 1

                conflict_3 = state
                conflict_3.x += 1
                conflict_3.t += 1

                conflict_4 = state
                conflict_4.x -= 1
                conflict_4.t += 1

                conflict_5 = state
                conflict_5.y += 1
                conflict_5.t += 1

                conflict_6 = state
                conflict_6.y -= 1
                conflict_6.t += 1

                conflict_7 = state
                conflict_7.x += 1
                conflict_7.y += 1
                conflict_7.t += 1

                conflict_8 = state
                conflict_8.x -= 1
                conflict_8.y += 1
                conflict_8.t += 1

                conflict_9 = state
                conflict_9.x -= 1
                conflict_9.y -= 1
                conflict_9.t += 1
                conflicts = [conflict_1, conflict_2, conflict_3, conflict_4, conflict_5, conflict_6, conflict_7, conflict_8, conflict_9]
            else:
                conflict_1 = state
                conflict_2 = state
                conflict_2.t += 1

                conflict_3 = state
                conflict_3.x += 1
                conflict_3.t += 1

                conflict_4 = state
                conflict_4.x -= 1
                conflict_4.t += 1

                conflict_5 = state
                conflict_5.y += 1
                conflict_5.t += 1

                conflict_6 = state
                conflict_6.y -= 1
                conflict_6.t += 1
                conflicts = [conflict_1, conflict_2, conflict_3, conflict_4, conflict_5]

            for c in conflicts:
                if c in val:
                    self.conflicts_found += 1
                    return True
        return False

    def __str__(self):
        return "Id: %d | Init: [%d;%d] | Goal: [%d;%d]" % (
            self.id,
            self.init_pos[0],
            self.init_pos[1],
            self.goal_pos[0],
            self.goal_pos[1],
        )
