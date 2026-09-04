"""Report recorded episode metrics without modifying rewards or delays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from msage_mappo.utils.config import parse_config_args

KEYS = ["domain", "scenario", "method"]
METRICS = {
    "episode_reward": "Mean step reward",
    "avg_delay_ms": "Reported delay (ms)",
    "violation_rate": "Logged risk metric",
}


def training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["seed", "episode", "episode_reward"])
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing EpisodeMetrics columns: {sorted(missing)}")
    keep = pd.Series(True, index=frame.index)
    if "aligned_start" in frame:
        keep &= pd.to_numeric(frame["aligned_start"], errors="coerce").fillna(0).eq(0)
    if "phase" in frame:
        keep &= frame["phase"].ne("aligned_neural_start")
    out = frame.loc[keep].copy()
    if out.empty:
        raise ValueError("No training episodes remain after excluding common-action diagnostics")
    if out[KEYS + ["seed", "episode"]].isna().any().any():
        raise ValueError("Missing run identifiers in EpisodeMetrics")
    for col in ("seed", "episode", *METRICS):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="raise")
            if np.isinf(out[col].to_numpy(dtype=float)).any():
                raise ValueError(f"Infinite values in {col}")
    if not np.isfinite(out["episode_reward"]).all():
        raise ValueError("Nonfinite recorded rewards; inspect the training run")
    if out.duplicated(KEYS + ["seed", "episode"]).any():
        raise ValueError("Duplicate method/seed/episode records; do not merge overlapping runs")
    return out.sort_values(KEYS + ["seed", "episode"]).reset_index(drop=True)


def summarize(frame: pd.DataFrame, last_window: int):
    tail = frame.groupby(KEYS + ["seed"], sort=False, group_keys=False).tail(last_window)
    records = []
    for identity, group in tail.groupby(KEYS + ["seed"], sort=False):
        record = dict(zip(KEYS + ["seed"], identity))
        record.update(episodes_used=len(group), first_episode=int(group.episode.min()), last_episode=int(group.episode.max()))
        for metric in METRICS:
            if metric in group and group[metric].notna().any():
                record[metric] = float(group[metric].mean())
        records.append(record)
    per_seed = pd.DataFrame(records)
    records = []
    for identity, group in per_seed.groupby(KEYS, sort=False):
        record = dict(zip(KEYS, identity))
        record.update(seeds=len(group), requested_window=last_window, min_episodes_used=int(group.episodes_used.min()))
        for metric in METRICS:
            if metric in group and group[metric].notna().any():
                record[metric + "_mean"] = float(group[metric].mean())
                record[metric + "_std_across_seeds"] = float(group[metric].std(ddof=1)) if group[metric].count() > 1 else np.nan
        records.append(record)
    return per_seed, pd.DataFrame(records)


def curve_points(frame: pd.DataFrame, metric: str, window: int) -> pd.DataFrame:
    frame = frame.sort_values(["seed", "episode"]).copy()
    frame["plotted"] = frame.groupby("seed")[metric].transform(
        lambda values: values.rolling(window, min_periods=1).mean()
    )
    return frame.groupby("episode")["plotted"].agg(["mean", "std", "count"]).reset_index()


def plot(frame: pd.DataFrame, output: Path, window: int):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    point_tables = []
    for domain, domain_rows in frame.groupby("domain", sort=True):
        if not str(domain).replace("_", "").isalnum():
            raise ValueError("Invalid domain identifier")
        metrics = [key for key in METRICS if key in domain_rows and domain_rows[key].notna().any()]
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.2), squeeze=False)
        identities = list(domain_rows.groupby(["scenario", "method"], sort=True))
        colors = plt.get_cmap("tab10")
        for idx, ((scenario, method), group) in enumerate(identities):
            for axis, metric in zip(axes[0], metrics):
                if not group[metric].notna().any():
                    continue
                points = curve_points(group, metric, window)
                label = str(method)
                if domain_rows.groupby("method")["scenario"].nunique().max() > 1:
                    label += " / " + str(scenario)
                color = colors(idx % 10)
                axis.plot(points.episode, points["mean"], label=label, color=color, linewidth=1.7)
                if points["count"].min() > 1:
                    axis.fill_between(points.episode, points["mean"] - points["std"],
                                      points["mean"] + points["std"], color=color, alpha=0.15, linewidth=0)
                axis.set(xlabel="Recorded training episode", ylabel=METRICS[metric])
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                axis.grid(True, color="0.88", linewidth=0.6)
                point_tables.append(points.assign(domain=domain, scenario=scenario, method=method,
                                                   metric=metric, trailing_window=window))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), fontsize=8, frameon=False)
        legend_height = 0.11 * max(1, int(np.ceil(len(labels) / 3)))
        fig.tight_layout(rect=(0, min(0.48, legend_height), 1, 1))
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{domain}_training_curves.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return pd.concat(point_tables, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Raw experiment workbooks with EpisodeMetrics")
    parser.add_argument("--output-dir", default="outputs/reports")
    parser.add_argument("--last-window", type=int, default=100)
    parser.add_argument("--smooth-window", type=int, default=1, help="Trailing display smoothing; raw data remains unchanged")
    args = parse_config_args(parser)
    frames, sources = [], []
    for filename in args.input:
        path = Path(filename)
        frame = pd.read_excel(path, sheet_name="EpisodeMetrics")
        frame["source_workbook"] = path.name
        frames.append(frame)
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    raw = training_rows(pd.concat(frames, ignore_index=True, sort=False))
    per_seed, summary = summarize(raw, args.last_window)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "episode_metrics.csv", index=False)
    per_seed.to_csv(output / "last_window_per_seed.csv", index=False)
    summary.to_csv(output / "last_window_summary.csv", index=False)
    plot(raw, output, args.smooth_window).to_csv(output / "plot_points.csv", index=False)
    protocol = {"sources": sources, "last_window": args.last_window, "smooth_window": args.smooth_window,
                "band": "one sample standard deviation across seeds; none for one seed",
                "excludes": "aligned common-action diagnostics", "metric_source": "recorded training episodes, not best-checkpoint evaluation"}
    (output / "report_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved raw-derived reports to {output}")


if __name__ == "__main__":
    main()
