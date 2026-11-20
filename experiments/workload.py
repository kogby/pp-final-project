"""Workload model from TraceLab: per-request shape, tool latency tail, headroom.

    uv run experiments/workload.py [--cap 1800] [--llm results/llm_rates/rates.csv]

Writes results/workload/*.csv and *.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agentsched.headroom import LocalLLM, capped, headroom
from agentsched.plot import SERIES, plt, save
from agentsched.tools import category
from agentsched.traces import ROOT, load_steps, tool_calls

OUT = ROOT / "results/workload"
CAPS = [None, 3600, 1800, 600, 300, 120]


def request_shape(steps: pd.DataFrame) -> pd.DataFrame:
    g = steps.groupby("request_id")
    per = pd.DataFrame({
        "provider": g.provider.first(),
        "steps": g.size(),
        "tool_calls": g.n_tool_calls.sum(),
        "ctx_end": g.ctx_tokens.max(),
        "out_tokens": g.output_tokens.sum(),
    })
    rows = []
    for prov, d in [("all", per), *per.groupby("provider")]:
        rows.append({
            "provider": prov,
            "requests": len(d),
            "steps_mean": d.steps.mean(),
            "tool_calls_mean": d.tool_calls.mean(),
            "no_tool_frac": (d.tool_calls == 0).mean(),
            "ctx_end_p50": d.ctx_end.median(),
            "ctx_end_p90": d.ctx_end.quantile(0.9),
            "out_tokens_p50": d.out_tokens.median(),
        })
    return pd.DataFrame(rows)


def tool_table(tc: pd.DataFrame) -> pd.DataFrame:
    tc = tc.assign(cat=tc.tool_name.map(category), s=tc.tool_wall_latency_ms / 1000)
    total = tc.s.sum()
    g = tc.groupby("cat").s
    out = pd.DataFrame({
        "calls": g.size(),
        "call_frac": g.size() / len(tc),
        "p50_s": g.median(),
        "p90_s": g.quantile(0.9),
        "p99_s": g.quantile(0.99),
        "time_frac": g.sum() / total,
    }).sort_values("calls", ascending=False)
    return out


def tail_figure(tc: pd.DataFrame) -> None:
    """Share of calls vs share of tool time: a few long calls dominate."""
    s = np.sort(tc.tool_wall_latency_ms.dropna().to_numpy() / 1000)[::-1]
    call_frac = np.arange(1, len(s) + 1) / len(s)
    time_frac = np.cumsum(s) / s.sum()
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(call_frac * 100, time_frac * 100, color=SERIES[0])
    k = np.searchsorted(s[::-1], 60, side="left")
    i = len(s) - k  # number of calls > 60 s
    ax.plot([call_frac[i - 1] * 100], [time_frac[i - 1] * 100], "o", color=SERIES[0])
    ax.annotate(f"calls > 60 s: {call_frac[i-1]:.1%} of calls,\n{time_frac[i-1]:.0%} of tool time",
                (call_frac[i - 1] * 100, time_frac[i - 1] * 100), xytext=(12, -28),
                textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("longest calls (% of all calls, log)")
    ax.set_ylabel("% of total tool time")
    ax.set_title("Tool time is concentrated in a few long calls")
    save(fig, OUT / "tool_tail.png")


def headroom_tables(steps: pd.DataFrame, cap: float, llms: list[LocalLLM]) -> pd.DataFrame:
    rows = []
    for c in CAPS:
        rows.append({"llm": "cloud (recorded)", "cap_s": c, **headroom(capped(steps, c))})
    st = capped(steps, cap)
    for m in llms:
        L = m.step_time(st.new_tokens, st.output_tokens)
        rows.append({"llm": f"{m.name} @{m.ctx // 1024}k", "cap_s": cap, **headroom(st, L)})
    return pd.DataFrame(rows)


def headroom_figure(h: pd.DataFrame) -> None:
    d = h[h.llm == "cloud (recorded)"].copy()
    d["cap"] = d.cap_s.map(lambda c: "none" if pd.isna(c) else f"{int(c) // 60} min")
    fig, ax = plt.subplots(figsize=(4.2, 3))
    x = np.arange(len(d))
    ax.plot(x, d.bound_one, "o-", label="one-step overlap", color=SERIES[0])
    ax.plot(x, d.bound_unl, "o-", label="unlimited lookahead", color=SERIES[1])
    ax.set_xticks(x, d.cap)
    ax.set_xlabel("idle cap on a single L or T gap")
    ax.set_ylabel("single-agent speedup bound (x)")
    ax.set_ylim(1, None)
    ax.legend()
    ax.set_title("Overlap within one agent has a low ceiling")
    save(fig, OUT / "headroom_vs_cap.png")


def load_llms(path: Path | None) -> list[LocalLLM]:
    if path is None or not path.exists():
        return []
    r = pd.read_csv(path)
    return [LocalLLM(x.model, int(x.ctx), x.prefill_tps, x.decode_tps) for x in r.itertuples()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=1800, help="idle cap (s) for the main table")
    ap.add_argument("--llm", type=Path, default=ROOT / "results/llm_rates/rates.csv")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    steps = load_steps()
    tc = tool_calls()

    shape = request_shape(steps)
    shape.to_csv(OUT / "request_shape.csv", index=False, float_format="%.3f")
    print(shape.round(2).to_string(index=False), "\n")

    tools = tool_table(tc)
    tools.to_csv(OUT / "tool_categories.csv", float_format="%.4f")
    print(tools.round(3).to_string(), "\n")
    tail_figure(tc)

    h = headroom_tables(steps, args.cap, load_llms(args.llm))
    h.to_csv(OUT / "headroom.csv", index=False, float_format="%.4f")
    print(h.round(3).to_string(index=False))
    headroom_figure(h)


if __name__ == "__main__":
    main()
