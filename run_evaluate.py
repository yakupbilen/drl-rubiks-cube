"""
For running: python run_evaluate.py -p "ini_files/solve.ini" -s 15,20,100 -t 10,20,60 -g 100
"""
import argparse
import torch
from networks.getNetwork import getNetwork
from evaluate.evaluate_model import evaluate
from datetime import datetime
from configs.solve_conf import SolveConfig
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tune gamma")
    parser.add_argument("-p","--ini_path",type=str,metavar="",required=True, help="Path of config file. Extension of file must be .ini")
    parser.add_argument("-g", "--games", type=str, metavar="", required=True,
                        help="Games per depth-threshold pairs.")
    parser.add_argument("-s", "--scramble_depths", type=str, metavar="", required=True,
                        help="Depths of cubes to be analyzed. Depths must be seperate by commas")
    parser.add_argument("-t", "--time_thresholds", type=str, metavar="", required=True,
                        help="Threshold times of cubes to be analyzed. Times must be seperate by commas")
    args = parser.parse_args()

    conf = SolveConfig(args.ini_path)

    net = getNetwork(conf.network_type)
    device = torch.device("cpu")
    net = net(54*6).to(device)
    net.load_state_dict(torch.load(conf.heuristic_path, map_location=device))
    games = int(args.games)
    scramble_depths = args.scramble_depths.split(',')
    scramble_depths = [int(item) for item in scramble_depths]
    max_times = args.time_thresholds.split(',')
    max_times = [int(item) for item in max_times]

    date_time = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

    evaluate(net=net,games=games,batch_size=conf.batch_size,
             weight=conf.weight,scramble_depths=scramble_depths,
             max_times=max_times, date_time=date_time)

