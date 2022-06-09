import sys
from search.search import AStar
from cube import cube
import matplotlib.pyplot as plt
import os


def search_weight_tuner(net, games, scramble_depth, max_iter, date_time):
    if not os.path.exists(f"analysis/search_parameters/{str(date_time)}"):
        os.makedirs(f"analysis/search_parameters/{str(date_time)}")
    sys.stdout = open(f"analysis/search_parameters/{str(date_time)}/lambda.txt", "w")
    weights = [i*0.01 for i in range(1,10)]
    weights.extend([i*0.1 for i in range(1,6)])
    states = []
    results = [[],[],[],[]]  # "accuracy", "mean_iteration", "mean_node", "mean_sol_len"
    metrics = ["Accuracy", "Mean Iteration", "Mean Generated Nodes", "Mean Solution Length"]
    metrics_file = ["accuracy", "mean_iter", "mean_node", "mean_sol"]
    for i in range(games):
        state,_ = cube.scramble(scramble_depth)
        states.append(state)

    for idx,weight in enumerate(weights):
        astar = AStar(weight=weight,max_iter=max_iter,max_time=None,batch_size=16)
        results[0].append(0)
        results[1].append(0)
        results[2].append(0)
        results[3].append(0)
        for state in states:
            solved,iteration,generated_nodes,solution = astar.search(state,net)
            if solved:
                results[0][idx] += 1
                results[1][idx] += generated_nodes
                results[2][idx] += iteration
                results[3][idx] += len(solution)

        results[1][idx] = results[1][idx]/results[0][idx] \
                                                if results[0][idx] else 0
        results[2][idx] = results[2][idx]/results[0][idx] \
                                                if results[0][idx] else 0
        results[3][idx] = results[3][idx]/results[0][idx] \
                                                if results[0][idx] else 0
        results[0][idx] /= games

        print(f"""Weight {weight} : 
        Accuracy : {results[0][idx]}__
        Mean Generated Nodes : {results[1][idx]}__
        Mean Iteration : {results[2][idx]}
        Mean Solution Length : {results[3][idx]}
                                """)

    sys.stdout.close()
    for result, metric, file_name in zip(results, metrics, metrics_file):
        fig, ax1 = plt.subplots(1, figsize=(30, 15))
        ax1.set_title(f"Games : {games} - Scramble Depth : {scramble_depth} - "
                      f"Max Iteration : {max_iter} ", fontsize=20, color="r")
        ax1.plot(weights, result, 'm')
        ax1.set_xticks(weights)
        ax1.tick_params(axis='x', labelsize=11)
        ax1.tick_params(axis='y', labelsize=15)
        plt.xticks(rotation=90)
        ax1.set_xlabel("Weights", fontsize=20, color="b")
        ax1.set_ylabel(metric, fontsize=20, color="b")
        plt.savefig(f"analysis/search_parameters/{str(date_time)}/lambda_{file_name}.png")
    return results


def search_batch_size_tuner(net, games, scramble_depth, max_time, date_time):
    if not os.path.exists(f"analysis/search_parameters/{str(date_time)}"):
        os.makedirs(f"analysis/search_parameters/{str(date_time)}")
    sys.stdout = open(f"analysis/search_parameters/{str(date_time)}/batch_size.txt", "w")
    batch_sizes = [i * 8 for i in range(1, 16)]

    states = []
    results = [[], [], []]  # "accuracy",, "mean_node", "mean_sol_len"
    metrics = ["Accuracy", "Mean Iteration", "Mean Generated Nodes", "Mean Solution Length"]
    metrics_file = ["accuracy", "mean_node", "mean_sol"]
    for i in range(games):
        state, _ = cube.scramble(scramble_depth)
        states.append(state)

    for idx,batch_size in enumerate(batch_sizes):
        astar = AStar(weight=0.05, max_iter=None, max_time=max_time, batch_size=batch_size)
        results[0].append(0)
        results[1].append(0)
        results[2].append(0)
        for state in states:
            solved, _, nodes, solution = astar.search(state, net)
            if solved:
                results[0][idx] += 1
                results[1][idx] += nodes
                results[2][idx] += len(solution)

        results[1][idx] = results[1][idx] / results[0][idx] \
            if results[0][idx] else 0
        results[2][idx] = results[2][idx] / results[0][idx] \
            if results[0][idx] else 0
        results[0][idx] /= games
        print(f"""Batch Size {batch_size} : 
        Accuracy : {results[0][idx]}__
        Mean Generated Nodes : {results[1][idx]}__
        Mean Solution Length : {results[2][idx]}
                                        """)
    sys.stdout.close()
    for result, metric, file_name in zip(results, metrics, metrics_file):
        fig, ax1 = plt.subplots(1, figsize=(30, 15))
        ax1.set_title(f"Games : {games} - Scramble Depth : {scramble_depth} - "
                      f"Max Time : {max_time}s ", fontsize=20, color="r")
        ax1.plot(batch_sizes, result, 'm')
        ax1.set_xticks(batch_sizes)
        ax1.tick_params(axis='x', labelsize=11)
        ax1.tick_params(axis='y', labelsize=15)
        plt.xticks(rotation=90)
        ax1.set_xlabel("Batch Size", fontsize=20, color="b")
        ax1.set_ylabel(metric, fontsize=20, color="b")
        plt.savefig(f"analysis/search_parameters/{str(date_time)}/batch_size_{file_name}.png")
    return results

