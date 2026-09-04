"""Orchestrate the five historical attribution variants with shared settings."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

from .full import parse_args
from msage_mappo.utils.config import REPO_ROOT

GROUPS = (
    ("qwen", ["full", "prior_only", "memory_no_refine", "random_memory"]),
    ("template", ["memory_no_refine"]),
)
NAMES = {
    "V2X-M-SAGE-full": "Hippo-full",
    "V2X-PriorOnly": "PriorOnly",
    "V2X-MemoryNoRefine": "Qwen-memory-only",
    "V2X-RandomMemory": "Random-memory",
}


def group_configs(args):
    if args.domains != ["v2x"]:
        raise ValueError("Attribution runner supports domains: [v2x] only")
    if args.llm_backend != "qwen" or args.allow_template_fallback:
        raise ValueError("Qwen attribution requires llm_backend: qwen and no fallback")
    parent = Path(args.output).parent
    groups = []
    for backend, methods in GROUPS:
        cfg = {key: deepcopy(value) for key, value in vars(args).items() if value is not None}
        cfg.pop("config", None)
        cfg.update(llm_backend=backend, v2x_methods=methods,
                   output=str(parent / f"{backend}_results.xlsx"))
        groups.append((backend, cfg))
    return groups


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dry-run", action="store_true", help="Show group configurations without training")
    preliminary, remaining = pre.parse_known_args()
    if "--help" in remaining:
        print("Additional ablation option: --dry-run (no model loading or training)")
    args = parse_args(remaining)
    groups = group_configs(args)
    if preliminary.dry_run:
        for backend, cfg in groups:
            print(f"# {backend} group\n{yaml.safe_dump(cfg, sort_keys=True)}")
        return
    if not Path(args.model_path).is_file():
        raise FileNotFoundError("Provide the Qwen GGUF file with --model-path before running this ablation")
    final = Path(args.output)
    targets = [final] + [Path(cfg["output"]) for _, cfg in groups]
    if any(path.exists() for path in targets):
        raise FileExistsError("Choose a fresh output directory for an attribution run")
    final.parent.mkdir(parents=True, exist_ok=True)
    combined = {}
    for backend, cfg in groups:
        config_path = final.parent / f"{backend}_resolved.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
        subprocess.run([
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_full_experiments.py"),
            "--config", str(config_path),
        ], cwd=REPO_ROOT, check=True)
        with pd.ExcelFile(cfg["output"]) as workbook:
            for sheet in workbook.sheet_names:
                frame = pd.read_excel(workbook, sheet_name=sheet)
                if "method" in frame:
                    frame["original_method"] = frame["method"]
                    mapping = dict(NAMES)
                    if backend == "template":
                        mapping["V2X-MemoryNoRefine"] = "Template-memory-only"
                    frame["method"] = frame["method"].replace(mapping)
                frame["source_group"] = backend
                combined.setdefault(sheet, []).append(frame)
    with pd.ExcelWriter(final, engine="openpyxl") as writer:
        for sheet, frames in combined.items():
            pd.concat(frames, ignore_index=True, sort=False).to_excel(writer, sheet_name=sheet, index=False)
    print(f"Saved {final}. Use plot_results.py --last-window 100 for windowed statistics.")


if __name__ == "__main__":
    main()
