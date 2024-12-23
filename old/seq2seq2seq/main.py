import argparse

from nn_expt.experiment import run_seq2seq2seq
from nn_expt.optimize import optimize
from nn_expt.utils import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    optimize(run_seq2seq2seq, config)
