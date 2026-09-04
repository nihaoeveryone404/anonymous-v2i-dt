"""Validated YAML defaults with explicit CLI overrides and portable paths."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PATH_FIELDS = {"v2x_env", "model_path", "output", "output_dir", "input"}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: str | Path, ancestors: tuple[Path, ...] = ()) -> dict:
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError(f"Circular config inheritance at {path.name}")
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Config must contain a YAML mapping")
    raw = dict(raw)
    parents = raw.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
        raise ValueError("extends must be a path or a list of paths")
    result = {}
    for parent in parents:
        result.update(load_config(path.parent / parent, ancestors + (path,)))
    normalized = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError("Config keys must be strings")
        dest = key.replace("-", "_")
        if dest in normalized:
            raise ValueError(f"Duplicate normalized config key: {dest}")
        normalized[dest] = value
    result.update(normalized)
    return result


def parse_config_args(parser: argparse.ArgumentParser, argv: list[str] | None = None):
    parser.add_argument("--config", help="YAML defaults, resolved from the repository root")
    parser.add_argument("--torch-threads", type=int, default=1)
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    selected, _ = pre.parse_known_args(argv)
    defaults = {}
    if selected.config:
        try:
            defaults = load_config(repo_path(selected.config))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            parser.error(str(exc))

    actions = {}
    for action in parser._actions:
        actions.setdefault(action.dest, []).append(action)
    unknown = set(defaults) - (set(actions) - {"help", "config"})
    if unknown:
        parser.error("Unknown config keys: " + ", ".join(sorted(unknown)))
    tokens = []
    for key, value in defaults.items():
        action = actions[key][0]
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(value, bool):
                parser.error(f"{key} must be a YAML boolean")
            parser.set_defaults(**{key: value})
            continue
        if value is None:
            parser.error(f"{key} must not be null; omit it to use the default")
        flag = next((x for x in action.option_strings if x.startswith("--")), None)
        if flag is None:
            parser.error(f"{key} cannot be supplied through config")
        multiple = action.nargs in ("*", "+") or isinstance(action.nargs, int)
        if multiple != isinstance(value, list):
            parser.error(f"{key} must be {'a list' if multiple else 'a scalar'}")
        values = value if multiple else [value]
        if any(isinstance(x, (dict, list, bool)) for x in values):
            parser.error(f"Invalid value for {key}")
        tokens.extend([flag, *[str(x) for x in values]])

    args = parser.parse_args(tokens + argv)
    for name in PATH_FIELDS:
        value = getattr(args, name, None)
        if value is not None:
            resolved = [str(repo_path(x)) for x in value] if isinstance(value, list) else str(repo_path(value))
            setattr(args, name, resolved)
    if args.config:
        args.config = str(repo_path(args.config))
    for name in ("episodes", "episode_len", "memory_every", "llm_every", "memory_dim", "top_k", "batch_size", "torch_threads", "last_window", "smooth_window"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    if getattr(args, "progress_every", 0) < 0:
        parser.error("progress_every must be nonnegative")
    if hasattr(args, "domains") and (not args.domains or set(args.domains) - {"v2x", "vmas"}):
        parser.error("domains must contain v2x and/or vmas")
    seeds = getattr(args, "seeds", None)
    if seeds is not None and (not seeds or len(set(seeds)) != len(seeds)):
        parser.error("seeds must be nonempty and unique")
    import torch
    torch.set_num_threads(args.torch_threads)
    return args
