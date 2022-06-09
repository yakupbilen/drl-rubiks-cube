import time
import torch
from search.node import Node
from search.search_utils import explore_states
from cube import cube
import operator
from keras.utils.np_utils import to_categorical
from math import floor


class AStar:
    def __init__(self,weight,max_iter,max_time,batch_size):
        self.weight = weight
        self.batch_size = batch_size
        if max_iter:
            self.max = max_iter
            self.search = self._search_by_iter
        else:
            self.max = max_time
            self.search = self._search_by_time

    def set_max_time(self,max_time):
        self.max = max_time
        self.search = self._search_by_time

    def set_max_iter(self,max_iter):
        self.max = max_iter
        self.search = self._search_by_iter

    def _search_by_iter(self,state,heuristic):
        if cube.is_solved(state) == 1:
            return True,0,0,[]
        open_nodes = []
        closed_nodes = {}
        start_node = Node(cube=state.copy(),depth=0,move=-1)
        open_nodes.append(start_node)
        generate_nodes = 1
        iteration = 0
        heuristic.eval()
        with torch.no_grad():
            while iteration < self.max:
                open_nodes.sort(key=operator.attrgetter('f'))
                current_nodes = [open_nodes.pop() for _ in range(
                                    min(len(open_nodes),self.batch_size))]
                current_states = [node.cube for node in current_nodes]
                new_states, goals = explore_states(current_states)
                if sum(goals) > 0:
                    ix = goals.index(1)
                    node_founded = current_nodes[floor(ix/12)]
                    solution = [ix % 12]
                    while True:
                        move = node_founded.move
                        if move != -1:
                            solution.append(move)
                            node_founded = node_founded.parent
                        else:
                            break
                    solution.reverse()
                    solution = cube.clear_seq(solution)
                    return True,iteration,generate_nodes+goals.index(1),solution
                generate_nodes += len(goals)
                new_nodes = []
                for i in range(len(new_states)):
                    parent = current_nodes[i]
                    closed_nodes[hash(current_nodes[i])] = parent
                    for j in range(len(new_states[i])):
                        depth = parent.depth+1
                        new_nodes.append(
                            Node(cube=new_states[i][j],depth=depth,move=j,parent=parent)
                        )
                idxs = []
                for i,node in enumerate(new_nodes):
                    if hash(node) in closed_nodes:
                        if closed_nodes[hash(node)].depth > node.depth:
                            in_closed = closed_nodes.pop(hash(node))
                            in_closed.depth = node.depth
                            in_closed.parent = node.parent
                            in_closed.move = node.move
                            idxs.append(i)
                    else:
                        idxs.append(i)

                if len(idxs)>0:
                    nodes_to_open = [new_nodes[idx] for idx in idxs]
                    states_to_open = [node.cube for node in nodes_to_open]
                    states_to_open = torch.from_numpy(to_categorical(states_to_open))
                    heuristics = heuristic(states_to_open)
                    for i in range(len(nodes_to_open)):
                        nodes_to_open[i].f = heuristics[i]-(self.weight*nodes_to_open[i].depth)

                    open_nodes.extend(nodes_to_open)
                    iteration += 1
        return False, iteration, generate_nodes, None

    def _search_by_time(self,state,heuristic,max_t=None):
        if max_t is None:
            max_t = self.max
        if cube.is_solved(state) == 1:
            return True,0,0,[]
        open_nodes = []
        closed_nodes = {}
        start_node = Node(cube=state.copy(),depth=0,move=-1)
        open_nodes.append(start_node)
        generate_nodes = 1
        iteration = 0
        start_t = time.perf_counter()
        heuristic.eval()
        with torch.no_grad():
            while (time.perf_counter()-start_t)<max_t:
                open_nodes.sort(key=operator.attrgetter('f'))
                current_nodes = [open_nodes.pop() for _ in range(
                                        min(len(open_nodes),self.batch_size))]
                current_states = [node.cube for node in current_nodes]
                new_states, goals = explore_states(current_states)
                if sum(goals) > 0:
                    ix = goals.index(1)
                    node_founded = current_nodes[floor(ix / 12)]
                    solution = [ix % 12]
                    while True:
                        move = node_founded.move
                        if move != -1:
                            solution.append(move)
                            node_founded = node_founded.parent
                        else:
                            break
                    solution = cube.clear_seq(solution)
                    solution.reverse()
                    return True, iteration, generate_nodes + goals.index(1), solution
                generate_nodes += len(goals)
                new_nodes = []
                for i in range(len(new_states)):
                    parent = current_nodes[i]
                    closed_nodes[hash(current_nodes[i])] = parent
                    for j in range(len(new_states[i])):
                        depth = parent.depth+1
                        new_nodes.append(
                            Node(cube=new_states[i][j],depth=depth,move=j,parent=parent)
                        )
                idxs = []
                for i,node in enumerate(new_nodes):
                    if hash(node) in closed_nodes:
                        if closed_nodes[hash(node)].depth > node.depth:
                            in_closed = closed_nodes.pop(hash(node))
                            in_closed.depth = node.depth
                            in_closed.parent = node.parent
                            in_closed.move = node.move
                            idxs.append(i)
                    else:
                        idxs.append(i)

                if len(idxs)>0:
                    nodes_to_open = [new_nodes[idx] for idx in idxs]
                    states_to_open = [node.cube for node in nodes_to_open]
                    states_to_open = torch.from_numpy(to_categorical(states_to_open))
                    heuristics = heuristic(states_to_open)
                    for i in range(len(nodes_to_open)):
                        nodes_to_open[i].f = heuristics[i]-(self.weight*nodes_to_open[i].depth)

                    open_nodes.extend(nodes_to_open)

                iteration += 1
        return False, i, generate_nodes, None


