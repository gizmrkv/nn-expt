import argparse

from nn_expt.experiment import run_seq2seq
from nn_expt.utils import wandb_sweep

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", "-s", type=str, default=None)
    parser.add_argument("--config_path", "-c", type=str, default=None)

    wandb_sweep(run_seq2seq, **vars(parser.parse_args()))
