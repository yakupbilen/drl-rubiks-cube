import numpy as np
"""
Solved state = 
[0 0 0 0 0 0 0 0 0 
 1 1 1 1 1 1 1 1 1 
 2 2 2 2 2 2 2 2 2 
 3 3 3 3 3 3 3 3 3 
 4 4 4 4 4 4 4 4 4 
 5 5 5 5 5 5 5 5 5] shape = 54,
The following array specifies which colored tiles with their indexes will be in which position after the actions(L,U...) are applied.

0:Up, 1:Left, 2:Front, 3:Right, 4:Back, 5:Down
6:Up', 7:Left', 8:Front', 9:Right', 10:Back', 11:Down'

For example: 
 First 9 tiles of our cube:               [2,1,5,1,0,4,0,2,5
 First 9 value of indexes changes for Up: [6,3,0,7,4,1,8,5,2
 First 9 tiles after applied Up action:   [0,1,2,2,0,1,5,4,5
 
"""
idxs = np.array([
        # 0:Up
        [6,3,0,7,4,1,8,5,2,18,19,20,12,13,14,15,16,17,27,28,29,21,22,23,24,25,26,
               36,37,38,30,31,32,33,34,35,9,10,11,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53],
        # 1:Left
        [44,1,2,41,4,5,38,7,8,15,12,9,16,13,10,17,14,11,0,19,20,3,22,23,6,25,26,
               27,28,29,30,31,32,33,34,35,36,37,51,39,40,48,42,43,45,18,46,47,21,49,50,24,52,53],
        # 2:Front
        [0,1,2,3,4,5,17,14,11,9,10,45,12,13,46,15,16,47,24,21,18,25,22,19,26,23,20,
               6,28,29,7,31,32,8,34,35,36,37,38,39,40,41,42,43,44,33,30,27,48,49,50,51,52,53],
        # 3:Right
        [0,1,20,3,4,23,6,7,26,9,10,11,12,13,14,15,16,17,18,19,47,21,22,50,24,25,53,
               33,30,27,34,31,28,35,32,29,8,37,38,5,40,41,2,43,44,45,46,42,48,49,39,51,52,36],
        # 4:Back
        [29,32,35,3,4,5,6,7,8,2,10,11,1,13,14,0,16,17,18,19,20,21,22,23,24,25,26,
               27,28,53,30,31,52,33,34,51,42,39,36,43,40,37,44,41,38,45,46,47,48,49,50,9,12,15],
        # 5:Down
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,42,43,44,18,19,20,21,22,23,15,16,17,
               27,28,29,30,31,32,24,25,26,36,37,38,39,40,41,33,34,35,51,48,45,52,49,46,53,50,47],
        # 6:Up'
        [2,5,8,1,4,7,0,3,6,36,37,38,12,13,14,15,16,17,9,10,11,21,22,23,24,25,26,
               18,19,20,30,31,32,33,34,35,27,28,29,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53],
        # 7:Left'
        [18,1,2,21,4,5,24,7,8,11,14,17,10,13,16,9,12,15,45,19,20,48,22,23,51,25,26,
               27,28,29,30,31,32,33,34,35,36,37,6,39,40,3,42,43,0,44,46,47,41,49,50,38,52,53],
        # 8:Front'
        [0,1,2,3,4,5,27,30,33,9,10,8,12,13,7,15,16,6,20,23,26,19,22,25,18,21,24,
               47,28,29,46,31,32,45,34,35,36,37,38,39,40,41,42,43,44,11,14,17,48,49,50,51,52,53],
        # 9:Right'
        [0,1,42,3,4,39,6,7,36,9,10,11,12,13,14,15,16,17,18,19,2,21,22,5,24,25,8,
               29,32,35,28,31,34,27,30,33,53,37,38,50,40,41,47,43,44,45,46,20,48,49,23,51,52,26],
        # 10:Back'
        [15,12,9,3,4,5,6,7,8,51,10,11,52,13,14,53,16,17,18,19,20,21,22,23,24,25,26,
               27,28,0,30,31,1,33,34,2,38,41,44,37,40,43,36,39,42,45,46,47,48,49,50,35,32,29],
        # 11:Down'
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,25,26,18,19,20,21,22,23,33,34,35,
               27,28,29,30,31,32,42,43,44,36,37,38,39,40,41,15,16,17,47,50,53,46,49,52,45,48,51]])

solved = (np.arange(0, 6, 1, dtype=np.ushort).reshape((6, 1)) +
          np.zeros((6, 9),dtype=np.ushort)).flatten()


def get_solved():
    return solved.copy()


def is_solved(other,penalty=-1,reward=1):
    if np.array_equal(solved,other):
        return reward
    return penalty


def scramble(times=6):
    state = get_solved()
    actions = np.random.randint(0, 11,times)
    for action in actions:
        state = state[idxs[action]]

    if is_solved(state) == 1:
        scramble(times)
    return state,actions


def scramble_with_list(state,seq):
    for act in seq:
        state = state[idxs[act]]
    return state


def get_solution(actions):
    solution = actions[::-1]
    solution = (solution+6)%12
    return solution


def clear_seq(seq):
    """
    Clear actions sequence, if there are unnecessary actions for example: R,R,R,R(same 4 moves in order)
    or transform R,R,R to R'
    :param seq: Actions sequence
    :return: Cleared actions sequence
    """
    idxs = []
    for i in range(len(seq)-1):
        if seq[i]+6==seq[i+1] or seq[i]-6==seq[i+1]:
            idxs.append(i)
    temp = -1
    deleted = 0
    for i,idx in enumerate(idxs):
        if idx!=temp:
            del seq[idx-deleted]
            del seq[idx-deleted]
            deleted+=2
            temp = idx+1
    return seq


if __name__=="__main__":
    print(f"Solved Cube Represantation:\n{solved}")
    print(f"Cube State Shape:\n{solved.shape}")