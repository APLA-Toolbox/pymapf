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
            if state in val:
                self.conflicts_found += 1
                return True
        return False

    def __str__(self):
        return "Id: %d | Init: [%d;%d] | Goal: [%d;%d]" % (self.id, self.init_pos[0], self.init_pos[1], self.goal_pos[0], self.goal_pos[1])
