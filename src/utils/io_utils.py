"""Configuration and filesystem helpers used by command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return data


def parse_args_with_config(parser: argparse.ArgumentParser):
    """Load YAML defaults first, then let explicit CLI arguments override them."""
    known, _ = parser.parse_known_args()
    config = load_yaml(known.config)
    valid = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - valid)
    if unknown:
        parser.error(f"unknown configuration key(s): {', '.join(unknown)}")
    parser.set_defaults(**config)
    return parser.parse_args()


def save_json(data: Any, path: str | Path) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def first_existing(paths: Iterable[str | Path]) -> Path | None:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None
