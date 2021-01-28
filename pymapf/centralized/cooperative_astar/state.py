class State:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.t == other.t

    def __hash__(self):
        return hash((self.x, self.y, self.t))

    def __str__(self):
        return "<Position: [%d; %d], Time: %d" % (self.x, self.y, self.t)
