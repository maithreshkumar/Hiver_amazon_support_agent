from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def env_or(value: str | None, environment_name: str) -> str:
    return os.getenv(environment_name, value or "")

