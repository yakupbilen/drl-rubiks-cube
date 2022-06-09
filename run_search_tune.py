"""
For running: python run_search_tune.py -p "ini_files/search_tune.ini"
"""
import argparse
import torch
from networks.getNetwork import getNetwork
from evaluate.search_tuner import search_weight_tuner, search_batch_size_tuner
from datetime import datetime
from configs.analysis_conf import AnalysisConfig
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tune weight and batch_size for A*")
    parser.add_argument("-p","--ini_path",type=str,metavar="",required=True, help="Path of config file. Extension of file must be .ini")
    args = parser.parse_args()

    conf = AnalysisConfig(args.ini_path)

    net = getNetwork(conf.network_type)
    device = torch.device("cpu")
    net = net(54*6).to(device)
    net.load_state_dict(torch.load(conf.heuristic_path, map_location=device))

    date_time = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

    search_weight_tuner(net, conf.games, conf.scramble_depth, conf.max_iter, date_time)
    search_batch_size_tuner(net, conf.games, conf.scramble_depth, conf.max_time, date_time)


