"""Small plotting helpers shared by inference and decision code."""

from __future__ import annotations

from pathlib import Path


def save_figure(fig, output_without_suffix: str | Path, *, tight: bool = True) -> None:
    output = Path(output_without_suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(output.with_suffix(".png"), dpi=300, **options)
    fig.savefig(output.with_suffix(".pdf"), **options)
