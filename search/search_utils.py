import numpy as np
from cube.cube import idxs,is_solved


def explore_states(states):
    states = np.stack(states)
    rewards = []
    states = np.repeat(np.expand_dims(states, axis=1), repeats=12, axis=1)
    states = np.take_along_axis(states, idxs[None, ...], 2)
    for state in states:
        rewards.extend(list(map(is_solved, state, [0]*len(state))))
    return states,rewards
