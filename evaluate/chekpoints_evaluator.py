from configs.analysis_conf import AnalysisConfig
from cube import cube
from search.search import AStar
from networks.getNetwork import getNetwork
import os
import torch
import sys
import matplotlib.pyplot as plt
from datetime import datetime


class CheckpointEvaluator:
    def __init__(self,config):
        assert isinstance(config, AnalysisConfig)
        self.config = config

    def models_evaluate(self):
        date_time = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        if not os.path.exists(f"analysis/checkpoints/{str(date_time)}"):
            os.makedirs(f"analysis/checkpoints/{str(date_time)}")
        sys.stdout = open(f"analysis/checkpoints/{str(date_time)}/checkpoints.txt", "w")
        device = torch.device('cpu')

        states = []
        results = [[], [], [], []]  # "accuracy", "mean_iteration", "mean_node", "mean_sol_len"
        metrics = ["Accuracy", "Mean Iteration", "Mean Generated Nodes", "Mean Solution Length"]
        metrics_file = ["accuracy","mean_iter","mean_node","mean_sol"]
        for i in range(self.config.games):
            state, _ = cube.scramble(self.config.scramble_depth)
            states.append(state)

        model_paths = []
        for file in os.listdir(self.config.heuristic_path):
            if file.endswith(".dat"):
                model_paths.append(file)
        checkpoint_number = -1
        for idx,ch in enumerate(model_paths[0]):
            try:
                int(ch)
                checkpoint_number = idx
                break
            except ValueError:
                pass
        string_part = model_paths[0][:checkpoint_number]
        sorted_paths = []
        for i in range(len(model_paths)):
            sorted_paths.append(int(model_paths[i][checkpoint_number:-4]))
        sorted_paths.sort()
        for idx,checkpoint_times in enumerate(sorted_paths):
            sorted_paths[idx] = f"{string_part}{checkpoint_times}.dat"

        solver = AStar(weight=self.config.weight,
                       max_iter=self.config.max_iter, max_time=None,
                       batch_size=self.config.batch_size)
        for idx,path in enumerate(sorted_paths):

            net = getNetwork(self.config.network_type)
            net = net(54 * 6).to(device)
            net.load_state_dict(torch.load(f"{self.config.heuristic_path}/{path}",map_location=device))

            results[0].append(0)
            results[1].append(0)
            results[2].append(0)
            results[3].append(0)
            for i in range(len(states)):
                solved, iteration, generated_nodes,solution = solver.search(states[i].copy(), net)
                if solved:
                    results[0][idx] += 1
                    results[1][idx] += generated_nodes
                    results[2][idx] += iteration
                    results[3][idx] += len(solution)
            results[1][idx] = results[1][idx] / results[0][idx] \
                if results[0][idx] else 0
            results[2][idx] = results[2][idx] / results[0][idx] \
                if results[0][idx] else 0
            results[3][idx] = results[3][idx] / results[0][idx] \
                if results[0][idx] else 0
            results[0][idx] /= self.config.games
            print(f"""{path} : 
                Accuracy : {results[0][idx]}__
                Mean Generated Nodes : {results[1][idx]}__
                Mean Iteration : {results[2][idx]}
                Mean Solution Length : {results[3][idx]}
                        """)
        sys.stdout.close()
        for result, metric, file_name in zip(results, metrics, metrics_file):
            fig, ax1 = plt.subplots(1, figsize=(30, 15))
            ax1.set_title(f"Games : {self.config.games} - Scramble Depth : {self.config.scramble_depth} - "
                          f"Batch Size : {self.config.batch_size} - Max Iteration : {self.config.max_iter} ", fontsize=20,color="r")
            ax1.plot(sorted_paths, result, 'm')
            ax1.set_xticks(sorted_paths)
            ax1.tick_params(axis='x', labelsize=11)
            ax1.tick_params(axis='y', labelsize=15)
            plt.xticks(rotation=90)
            ax1.set_xlabel("Checkpoints", fontsize=20, color="b")
            ax1.set_ylabel(metric, fontsize=20, color="b")
            plt.savefig(f"analysis/checkpoints/{str(date_time)}/checkpoints_{file_name}_{str(date_time)}.png")
        return results

