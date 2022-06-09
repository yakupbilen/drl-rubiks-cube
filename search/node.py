import numpy as np


class Node:
    __slots__ = ["f","cube","parent","depth","move","hash"]

    def __init__(self,cube,depth,move,parent=None):
        self.cube = cube
        self.parent = parent
        self.depth = depth
        self.f = 0
        self.move = move
        try:
            self.hash = hash(self.cube.flatten().tobytes())
        except AttributeError:
            print(self.cube)

    def __eq__(self, other):
        return np.array_equal(self.cube.flatten(),other.cube.flatten())

    """def __lt__(self, other):
        return self.f > other.f"""

    def __hash__(self):
        return self.hash
