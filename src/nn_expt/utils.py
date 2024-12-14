import datetime
import json
from pathlib import Path
from typing import Any, Dict

import toml
import yaml


def get_run_name() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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
