"""
For running: python run_checkpoints_evaluate.py -p "ini_files/analysis.ini"
"""
from evaluate.chekpoints_evaluator import CheckpointEvaluator
from configs.analysis_conf import AnalysisConfig
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument("-p", "--ini_path", type=str, metavar="", required=True, help="Path of config file. Extension of file must be .ini")
    args = parser.parse_args()

    conf = AnalysisConfig(args.ini_path)
    evaluator = CheckpointEvaluator(conf)
    evaluator.models_evaluate()