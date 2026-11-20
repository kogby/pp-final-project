"""Gantt timeline of a replay log: one row per agent, prefill/decode/tool/wait.

    uv run experiments/timeline.py results/scaling/<host>/log_naive_N4.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from matplotlib.patches import Patch

from agentsched.plot import INK2, SERIES, plt, save

COLORS = {"prefill": SERIES[6], "decode": SERIES[0], "tool": SERIES[1], "wait": "#c3c2b7"}


def timeline(log: pd.DataFrame, title: str):
    agents = sorted(log.agent.unique())
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(agents) + 1.2))
    for kind, c in COLORS.items():
        d = log[log.kind == kind]
        for a in agents:
            x = d[d.agent == a]
            ax.broken_barh(list(zip(x.t0, x.t1 - x.t0)), (a - 0.35, 0.7), color=c, linewidth=0)
    for r in log[log.kind == "request"].itertuples():
        ax.plot([r.t1, r.t1], [r.agent - 0.45, r.agent + 0.45], color=INK2, lw=1)
    ax.set_yticks(agents, [f"agent {a}" for a in agents])
    ax.invert_yaxis()
    ax.set_xlabel("time (s)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in COLORS.items()],
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25 if len(agents) < 4 else -0.12))
    return fig


if __name__ == "__main__":
    for p in map(Path, sys.argv[1:]):
        save(timeline(pd.read_csv(p), p.stem), p.with_suffix(".png"))
