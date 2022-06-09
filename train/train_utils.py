import numpy as np
import torch
from random import randint
from keras.utils.np_utils import to_categorical
from cube.cube import get_solved,idxs


def generate_states_by_ADI(conf, nnet, device):
    times = conf.batch_size//conf.scramble_depth
    states, depths = generate_move_seq(times,conf.scramble_depth,conf.include_first)

    val_targets = explore_states(states,nnet,device).to(device)
    states = torch.from_numpy(to_categorical(states)).to(device)
    weights = get_weigths(depths,conf.weight_type).to(device)
    return states.detach(),val_targets.detach(),weights.detach()


def generate_move_seq(times,scramble_depth,include_first):
    seq = np.random.randint(12,size=times*scramble_depth)
    for t in range(times):
        for idx in range(1,scramble_depth):
            while True:
                if (seq[idx+t*scramble_depth] + 6) % 12 == seq[idx+t*scramble_depth - 1]:
                    seq[idx+t*scramble_depth] = randint(0, 11)
                elif idx > 1 and (seq[idx+t*scramble_depth] == seq[idx+t*scramble_depth - 2] and seq[idx+t*scramble_depth] == seq[idx+t*scramble_depth - 1]):
                    seq[idx+t*scramble_depth] = randint(0, 11)
                else:
                    break

    depths = np.ones(times*scramble_depth,dtype=np.float32)
    state = get_solved()
    states = np.repeat(np.expand_dims(state, axis=0), repeats=times*scramble_depth, axis=0)

    for time in range(times):
        states[time*scramble_depth] = states[time*scramble_depth][idxs[seq[time*scramble_depth]]]
        depths[time*scramble_depth] = 1
        for depth in range(scramble_depth-1):
            states[time * scramble_depth + depth + 1] = states[time * scramble_depth + depth][idxs[seq[time * scramble_depth + depth + 1]]]
            depths[time * scramble_depth + depth + 1] = depth+2

    if include_first:
        solved_states = np.repeat(np.expand_dims(state, axis=0), repeats=times, axis=0)
        solved_depths = np.ones(times,dtype=np.float32)
        states = np.concatenate([states,solved_states])
        depths = np.concatenate([depths,solved_depths])
    shuffle_idxs = np.arange(depths.shape[0])
    np.random.shuffle(shuffle_idxs)
    states = states[shuffle_idxs]
    depths = depths[shuffle_idxs]
    return states,depths


def generate_move_seq_random(times,scramble_depth,include_first):
    seq = np.random.randint(12,size=times*scramble_depth)
    depths = np.ones(times*scramble_depth,dtype=np.float32)
    state = get_solved()
    states = np.repeat(np.expand_dims(state, axis=0), repeats=times*scramble_depth, axis=0)

    for time in range(times):
        states[time*scramble_depth] = states[time*scramble_depth][idxs[seq[time*scramble_depth]]]
        depths[time*scramble_depth] = 1
        for depth in range(scramble_depth-1):
            states[time * scramble_depth + depth + 1] = states[time * scramble_depth + depth][idxs[seq[time * scramble_depth + depth + 1]]]
            depths[time * scramble_depth + depth + 1] = depth+2

    if include_first:
        solved_states = np.repeat(np.expand_dims(state, axis=0), repeats=times, axis=0)
        solved_depths = np.ones(times,dtype=np.float32)
        states = np.concatenate([states,solved_states])
        depths = np.concatenate([depths,solved_depths])
    shuffle_idxs = np.arange(depths.shape[0])
    np.random.shuffle(shuffle_idxs)
    states = states[shuffle_idxs]
    depths = depths[shuffle_idxs]
    return states,depths


def explore_states(states,net,device):
    solved = get_solved()
    substates = np.repeat(np.expand_dims(states, axis=1), repeats=12, axis=1)
    substates = np.squeeze(substates[:, ::12])[:, idxs]

    solved_states = (states == solved).all(axis=states.ndim-1)
    solved_substates = (substates == solved).all(axis=substates.ndim-1)
    rewards_substates = np.ones(solved_substates.shape,dtype=np.float32)
    rewards_substates[~solved_substates] = -1
    rewards_substates = torch.from_numpy(rewards_substates).to(device)
    substates = torch.from_numpy(to_categorical(substates)).to(device)
    values = net(substates)
    values = values.view(-1, 12) + rewards_substates.view(-1, 12)
    values = torch.max(values, 1)[0]
    values[solved_states] = 0

    return values


def update_generator(generator, nnet, tau, device):
    generator_params, net_params = generator.state_dict(), nnet.state_dict()
    new_generator_params = dict(generator_params)
    for name, param in net_params.items():
        new_generator_params[name].data.copy_(
            tau * param.data.to(device) + (1 - tau) * new_generator_params[name].data.to(device)
        )
    generator.load_state_dict(new_generator_params)
    return generator.to(device)


def get_weigths(depths, type):
    if type == "learn_first_close":
        return 1/torch.from_numpy(depths)
    elif type == "lightweight":
        return 1/(torch.from_numpy(depths)/2)
    else:
        return torch.from_numpy(depths)






