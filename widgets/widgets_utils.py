from dataclasses import dataclass
import numpy as np
from cube import cube
COLOR_VALUE = {"blue":0,"orange":1,"white":2,"red":3,"yellow":4,"green":5}
VALUE_COLOR = ["blue","orange","white","red","yellow","green"]
ACTIONS = ["U", "L", "F", "R", "B", "D", "U'", "L'", "F'", "R'", "B'", "D'"]


def seq_to_string(seq):
    global ACTIONS
    string_seq = ""
    for action in seq:
        string_seq = string_seq+" "+ACTIONS[action]
    return string_seq


def cubies_to_state(cubies):
    global COLOR_VALUE
    state = np.zeros((54,),dtype=np.ushort)
    for idx,cubie in enumerate(cubies):
        state[idx] = COLOR_VALUE[cubie[1]]
    return state

def scramble(cubies,depth=None,seq=None):
    global VALUE_COLOR
    if depth is None:
        seq = seq
    else:
        seq = np.random.randint(0, 12, depth)
    state = cubies_to_state(cubies)
    states = []
    for action in seq:
        state = state[cube.idxs[action]]
        states.append(state.copy())
    return states,seq


@dataclass()
class Settings:
    search_weight: float
    search_batch_size: int
    actions_delay: int = 250
    def_scramble_depth: int = 15
    def_time: int = 30
    def_iter: int = 100

