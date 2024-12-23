import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping

import optuna
import optuna.storages.journal
import optunahub
import toml
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == ".json":
        with path.open() as f:
            return json.load(f)
    elif path.suffix in [".yaml", ".yml"]:
        with path.open() as f:
            return yaml.safe_load(f)
    elif path.suffix == ".toml":
        return toml.load(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    merged = {}
    for d in dicts:
        duplicated_keys = set(merged.keys()) & set(d.keys())
        if duplicated_keys:
            raise ValueError(f"Duplicated keys: {duplicated_keys}")

        merged |= d
    return merged


def is_valid_clause(
    target: Mapping[str, Any], *required: str, optional: Iterable[str] | None = None
) -> bool:
    optional_keys = set(optional) if optional is not None else set()

    if not all(key in target for key in required):
        return False

    target_keys = set(target.keys())
    valid_keys = set(required) | optional_keys
    return target_keys.issubset(valid_keys)


def parse_grid_search_space(
    search_space: Mapping[str, Mapping[str, Any]],
) -> Dict[str, List[Any]]:
    grid_space: Dict[str, List[Any]] = {}
    for var_name, var_space in search_space.items():
        if is_valid_clause(var_space, "values"):
            values = var_space["values"]
            if isinstance(values, dict):
                grid_space |= {
                    f"{var_name}_{k}": v
                    for k, v in parse_grid_search_space(values).items()
                }
            elif isinstance(values, (list, tuple)):
                grid_space[var_name] = list(values)
        elif is_valid_clause(var_space, "min", "max"):
            min_v = var_space["min"]
            max_v = var_space["max"]
            if isinstance(min_v, int) and isinstance(max_v, int):
                grid_space[var_name] = list(range(min_v, max_v + 1))
            else:
                raise ValueError(
                    f"Invalid configuration for '{var_name}': 'min' and 'max' must be integers."
                )

    return grid_space


def embed_search_space(
    search_space: Dict[str, Any], base_dir: str | Path = "."
) -> Dict[str, Any]:
    base_dir = Path(base_dir)

    if "$path" in search_space:
        path = search_space["$path"]
        if not isinstance(path, str):
            raise ValueError(
                f"Invalid configuration for '$path': must be a string. Got {path}"
            )

        file_path = base_dir / path
        loaded_space = embed_search_space(load_config(file_path), file_path.parent)
        search_space.pop("$path")
        search_space = merge_dicts(loaded_space, search_space)
        return search_space

    elif "$paths" in search_space:
        paths = search_space["$paths"]
        if not isinstance(paths, (list, tuple)):
            raise ValueError(
                f"Invalid configuration for '$paths': must be a list or a tuple. Got {paths}"
            )

        file_paths = [base_dir / p for p in paths]
        merged_dict = merge_dicts(
            *[embed_search_space(load_config(fp), fp.parent) for fp in file_paths]
        )

        search_space.pop("$paths")
        search_space = merge_dicts(merged_dict, search_space)
        return search_space

    else:
        return search_space


def suggest_sample(
    trial: optuna.Trial, search_space: Mapping[str, Mapping[str, Any]], prefix: str = ""
) -> Dict[str, Any]:
    sample: Dict[str, Any] = {}
    for var_name, var_space in search_space.items():
        if is_valid_clause(var_space, "value"):
            sample[var_name] = var_space["value"]

        elif is_valid_clause(var_space, "values"):
            values = var_space["values"]

            if isinstance(values, dict):
                sample[var_name] = suggest_sample(
                    trial, values, prefix=prefix + var_name + "/"
                )

            elif isinstance(values, (list, tuple)):
                sample[var_name] = trial.suggest_categorical(prefix + var_name, values)

            else:
                raise ValueError(
                    f"Invalid configuration for '{var_name}': 'values' must be a list or a dictionary. Got {values}"
                )

        elif is_valid_clause(var_space, "min", "max", optional=["log", "step"]):
            min_v = var_space["min"]
            max_v = var_space["max"]

            if isinstance(min_v, int) and isinstance(max_v, int):
                step = var_space.get("step", 1)
                sample[var_name] = trial.suggest_int(
                    prefix + var_name, min_v, max_v, step=step
                )

            elif isinstance(min_v, float) and isinstance(max_v, float):
                log = var_space.get("log", False)
                step = var_space.get("step", None)
                sample[var_name] = trial.suggest_float(
                    prefix + var_name, min_v, max_v, log=log, step=step
                )

            else:
                raise ValueError(
                    f"Invalid configuration for '{var_name}': 'min' and 'max' must be of the same type (both int or both float). Got {min_v} and {max_v}"
                )

        else:
            raise ValueError(
                f"Invalid configuration for '{var_name}': must contain 'value', 'values', or 'min' and 'max'. Got {var_space}"
            )

    return sample


def optimize(
    objective: Callable[..., Any],
    config_path: Path | str,
    *,
    journal_file: str = "./journal.log",
    storage: optuna.storages.BaseStorage | None = None,
):
    config_path = Path(config_path)
    config = load_config(config_path)

    sampler_type = config.get("sampler", None)
    sampler_kwargs = config.get("sampler_kwargs", {})
    pruner_type = config.get("pruner", None)
    pruner_kwargs = config.get("pruner_kwargs", {})
    study_name = config.get("study_name", None)
    direction = config.get("direction", None)
    load_if_exists = config.get("load_if_exists", False)
    n_trials = config.get("n_trials", None)
    timeout = config.get("timeout", None)
    n_jobs = config.get("n_jobs", None)
    gc_after_trial = config.get("gc_after_trial", False)
    show_progress_bar = config.get("show_progress_bar", False)

    search_space = config.get("search_space", {})
    search_space = embed_search_space(search_space, base_dir=config_path.parent)

    sampler_types = {
        "grid": optuna.samplers.GridSampler,
        "random": optuna.samplers.RandomSampler,
        "tpe": optuna.samplers.TPESampler,
        "cmaes": optuna.samplers.CmaEsSampler,
        "gp": optuna.samplers.GPSampler,
        "partialfixed": optuna.samplers.PartialFixedSampler,
        "nsgaii": optuna.samplers.NSGAIISampler,
        "nsgaiii": optuna.samplers.NSGAIIISampler,
        "qmc": optuna.samplers.QMCSampler,
        "bruteforce": optuna.samplers.BruteForceSampler,
        "auto": optunahub.load_module("samplers/auto_sampler").AutoSampler,
    }
    sampler_type = sampler_types.get(sampler_type, None)
    if sampler_type == optuna.samplers.GridSampler:
        sampler_kwargs["search_space"] = parse_grid_search_space(search_space)
    sampler = sampler_type(**sampler_kwargs) if sampler_type else None

    pruner_types = {
        "median": optuna.pruners.MedianPruner,
        "nop": optuna.pruners.NopPruner,
        "patient": optuna.pruners.PatientPruner,
        "percentile": optuna.pruners.PercentilePruner,
        "successivehalving": optuna.pruners.SuccessiveHalvingPruner,
        "hyperband": optuna.pruners.HyperbandPruner,
        "threshold": optuna.pruners.ThresholdPruner,
        "wilcoxon": optuna.pruners.WilcoxonPruner,
    }
    pruner_type = pruner_types.get(pruner_type, None)
    pruner = pruner_type(**pruner_kwargs) if pruner_type else None

    storage = storage or optuna.storages.journal.JournalStorage(
        optuna.storages.journal.JournalFileBackend(journal_file)
    )
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        direction=direction,
        load_if_exists=load_if_exists,
    )

    def wrap_objective(
        objective: Callable[..., Any],
        search_space: Mapping[str, Mapping[str, Any]],
    ) -> Callable[[optuna.Trial], Any]:
        def _objective(trial: optuna.Trial) -> Any:
            return objective(**suggest_sample(trial, search_space), trial=trial)

        return _objective

    study.optimize(
        wrap_objective(objective, search_space),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        gc_after_trial=gc_after_trial,
        show_progress_bar=show_progress_bar,
    )
