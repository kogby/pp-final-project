"""Measure local LLM rates with llama.cpp's benchmark tools.

    uv run experiments/llm_rates.py [--model data/models/Qwen3-8B-Q4_K_M.gguf] [--quick]

1. llama-bench: prefill and decode tok/s at several context depths (one sequence).
2. llama-batched-bench: decode step time vs batch size N and per-sequence context,
   i.e. how each added agent's KV cache slows every agent's step.
Writes results/llm_rates/{depth,batched,rates}.csv and figures, tagged by host.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from agentsched.plot import SERIES, plt, save
from agentsched.profile import host
from agentsched.traces import ROOT

OUT = ROOT / "results/llm_rates"
MODEL = ROOT / "data/models/Qwen3-8B-Q4_K_M.gguf"


def jsonl(cmd: list[str]) -> list[dict]:
    print("$", " ".join(cmd), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.startswith("{")]


def depth_sweep(model: Path, depths: list[int], reps: int) -> pd.DataFrame:
    d = ",".join(map(str, depths))
    rows = jsonl(["llama-bench", "-m", str(model), "-p", "512", "-n", "128", "-d", d,
                  "-r", str(reps), "-fa", "on", "-o", "jsonl"])
    df = pd.DataFrame(rows)
    df["phase"] = df.n_prompt.map(lambda p: "prefill" if p > 0 else "decode")
    return df[["phase", "n_depth", "avg_ts", "stddev_ts"]].rename(columns={"n_depth": "ctx"})


def batched_sweep(model: Path, ctxs: list[int], npl: list[int]) -> pd.DataFrame:
    rows = []
    for c in ctxs:
        total = (c + 128) * max(npl) + 512
        rows += jsonl(["llama-batched-bench", "-m", str(model), "-c", str(total), "-fa", "on",
                       "-npp", str(c), "-ntg", "128", "-npl", ",".join(map(str, npl)),
                       "--output-format", "jsonl"])
    df = pd.DataFrame(rows)
    # t_tg: seconds to generate ntg tokens for all pl sequences -> per-step time.
    df["step_ms"] = df.t_tg / df.tg * 1000
    df["agg_tps"] = df.speed_tg
    return df[["n_kv_max", "pp", "pl", "step_ms", "agg_tps", "speed_pp"]].rename(
        columns={"pp": "ctx", "pl": "batch"})


def rates_table(depth: pd.DataFrame, name: str) -> pd.DataFrame:
    """LocalLLM rows: rates at mean depth ctx/2 for each modeled context cap."""
    piv = depth.pivot_table(index="ctx", columns="phase", values="avg_ts")
    rows = []
    for cap in (16384, 32768, 65536):
        mid = cap // 2
        if mid > piv.index.max():
            continue
        r = piv.iloc[(piv.index.to_series() - mid).abs().argmin()]
        rows.append({"model": name, "ctx": cap, "prefill_tps": r.prefill, "decode_tps": r.decode})
    return pd.DataFrame(rows)


def figures(depth: pd.DataFrame, batched: pd.DataFrame, tag: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, ph, c in zip(axes, ["prefill", "decode"], SERIES):
        d = depth[depth.phase == ph]
        ax.plot(d.ctx / 1024, d.avg_ts, "o-", color=c)
        ax.set_title(f"{ph} rate vs context")
        ax.set_xlabel("context depth (k tokens)")
        ax.set_ylabel("tokens/s")
        ax.set_ylim(0, None)
    save(fig, OUT / f"depth_{tag}.png")

    fig, ax = plt.subplots(figsize=(4.4, 3))
    for (ctx, d), c in zip(batched.groupby("ctx"), SERIES):
        ax.plot(d.batch, d.step_ms, "o-", color=c, label=f"{ctx // 1024}k ctx/agent")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("agents decoding in the batch (N)")
    ax.set_ylabel("decode step time (ms)")
    ax.set_ylim(0, None)
    ax.set_title("Each agent's KV cache slows every step")
    ax.legend()
    save(fig, OUT / f"batched_{tag}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--quick", action="store_true", help="fewer points, for a smoke test")
    ap.add_argument("--only", choices=["depth", "batched"], help="run one part; reuse saved CSV")
    a = ap.parse_args()

    depths = [0, 4096, 16384] if a.quick else [0, 2048, 4096, 8192, 16384, 32768]
    ctxs = [1024, 8192] if a.quick else [1024, 4096, 8192, 16384]
    npl = [1, 2, 4] if a.quick else [1, 2, 4, 8, 16]
    tag = host()
    name = a.model.stem

    OUT.mkdir(parents=True, exist_ok=True)
    if a.only == "batched":
        depth = pd.read_csv(OUT / f"depth_{tag}.csv")
    else:
        depth = depth_sweep(a.model, depths, reps=2 if a.quick else 3).assign(host=tag, model=name)
        depth.to_csv(OUT / f"depth_{tag}.csv", index=False)
        print(depth.round(1).to_string(index=False))
        rates_table(depth, name).assign(host=tag).to_csv(OUT / "rates.csv", index=False)

    if a.only == "depth":
        batched = pd.read_csv(OUT / f"batched_{tag}.csv")
    else:
        batched = batched_sweep(a.model, ctxs, npl).assign(host=tag, model=name)
        batched.to_csv(OUT / f"batched_{tag}.csv", index=False)
        print(batched.round(1).to_string(index=False))
    figures(depth, batched, tag)


if __name__ == "__main__":
    main()
