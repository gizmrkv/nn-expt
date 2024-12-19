import datetime
import json
from pathlib import Path
from typing import Any, Dict

import toml
import yaml


def get_run_name() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
