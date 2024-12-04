import datetime
from typing import Any, Callable

import wandb
import yaml


def get_run_name() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def wandb_sweep(
    function: Callable[[], Any],
    *,
    sweep_id: str | None = None,
    config_path: str | None = None,
    entity: str | None = None,
    project: str | None = None,
    count: int | None = None,
):
    if sweep_id is None and config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        sweep_id = wandb.sweep(config, project="seq2seq2seq")

    if sweep_id is None:
        raise ValueError("Wrong sweep_id or config_path")

    wandb.agent(
        sweep_id, function=function, entity=entity, project=project, count=count
    )
