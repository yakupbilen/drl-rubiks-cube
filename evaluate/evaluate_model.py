import sys
from search.search import AStar
from cube import cube
import os


def evaluate(net, games, batch_size, weight, scramble_depths, max_times, date_time):
    if not os.path.exists(f"analysis/result/{str(date_time)}"):
        os.makedirs(f"analysis/result/{str(date_time)}")
    sys.stdout = open(f"analysis/result/{str(date_time)}/result.txt", "w")
    states = []
    results = [[],[],[],[]]  # "accuracy", "mean_iteration", "mean_node", "mean_sol_len"
    for depth in scramble_depths:
        temp = []
        for i in range(games):
            state,_ = cube.scramble(depth)
            temp.append(state)
        states.append(temp)
    for idx,depth in enumerate(scramble_depths):
        for time_idx,time in enumerate(max_times):
            astar = AStar(weight=weight,max_iter=None,max_time=time,batch_size=batch_size)
            results[0].append(0)
            results[1].append(0)
            results[2].append(0)
            results[3].append(0)
            curr_idx = idx*len(max_times)+time_idx
            for state in states[idx]:
                solved,iteration,generated_nodes,solution = astar.search(state,net)
                if solved:
                    results[0][curr_idx] += 1
                    results[1][curr_idx] += generated_nodes
                    results[2][curr_idx] += iteration
                    results[3][curr_idx] += len(solution)

            results[1][curr_idx] = results[1][curr_idx]/results[0][curr_idx] \
                                                    if results[0][curr_idx] else 0
            results[2][curr_idx] = results[2][curr_idx]/results[0][curr_idx] \
                                                    if results[0][curr_idx] else 0
            results[3][curr_idx] = results[3][curr_idx]/results[0][curr_idx] \
                                                    if results[0][curr_idx] else 0
            results[0][curr_idx] /= games

            print(f"""Scramble Depth : {depth} - Time Threshold : {time}: 
                Accuracy : {results[0][curr_idx]}__
                Mean Generated Nodes : {results[1][curr_idx]}__
                Mean Iteration : {results[2][curr_idx]}
                Mean Solution Length : {results[3][curr_idx]}
                                    """)

    sys.stdout.close()
    return results
