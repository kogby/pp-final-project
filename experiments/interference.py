"""Tool data cards: two-way interference between each tool and batched decode.

    uv run experiments/interference.py --tools sleep spin stream pandas --batch 1 4

For each decode batch size B, B "hog" agents decode continuously on llama-server.
We record the decode rate with no tool, then run each tool and record
  tool_slowdown   = tool wall under decode / tool wall alone
  decode_slowdown = decode tok/s alone / decode tok/s while the tool runs
Writes results/interference/<host>/cards.csv and a figure.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import numpy as np
import pandas as pd

from agentsched.llm import Client, server
from agentsched.plot import SERIES, plt, save
from agentsched.profile import host, profile
from agentsched.toolsuite import TOOLS
from agentsched.traces import ROOT

CTX = 2048      # per-hog context
CHUNK = 16      # tokens per request, so rate samples are fine-grained


class Hogs:
    """B agents decoding back to back; records (t_end, tokens, decode_ms) samples."""

    def __init__(self, url: str, b: int):
        self.c = Client(url)
        self.b = b
        self.samples: list[tuple[float, float, int]] = []
        self.stop = False

    async def _one(self, slot: int) -> None:
        rng = np.random.default_rng(slot)
        ctx = rng.integers(1000, 30000, CTX).tolist()
        while not self.stop:
            t0 = time.perf_counter()
            r = await self.c.complete(ctx, CHUNK, slot=slot)
            self.samples.append((t0, time.perf_counter(), r["n_decode"]))

    def rate(self, t0: float, t1: float) -> float:
        """Aggregate decode tok/s over requests fully inside [t0, t1]."""
        s = [(a, b, n) for a, b, n in self.samples if a >= t0 and b <= t1]
        return sum(n for *_, n in s) / (t1 - t0) if s else float("nan")

    async def run(self) -> None:
        await asyncio.gather(*(self._one(i) for i in range(self.b)))
        await self.c.close()


async def measure(url: str, b: int, tools: list[str], warm: float, base: float) -> list[dict]:
    hogs = Hogs(url, b)
    task = asyncio.create_task(hogs.run())
    await asyncio.sleep(warm)
    rows = []
    for name in tools:
        t = TOOLS[name]
        b0 = time.perf_counter()
        await asyncio.sleep(base)
        r_alone = hogs.rate(b0, time.perf_counter())
        t0 = time.perf_counter()
        p = await asyncio.to_thread(profile, t.argv, t.cwd)
        t1 = time.perf_counter()
        rows.append({"tool": name, "batch": b, "tool_wall_s": p.wall_s,
                     "decode_tps_alone": r_alone, "decode_tps_with_tool": hogs.rate(t0, t1)})
        print(rows[-1], flush=True)
    hogs.stop = True
    await task
    return rows


def figure(cards: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for (b, d), c in zip(cards.groupby("batch"), SERIES):
        ax.scatter(d.tool_slowdown, d.decode_slowdown, s=40, color=c, label=f"B={b} decoding",
                   edgecolor="white", linewidth=1.5, zorder=3)
        for r in d.itertuples():
            ax.annotate(r.tool, (r.tool_slowdown, r.decode_slowdown), xytext=(5, 3),
                        textcoords="offset points", fontsize=8)
    ax.axhline(1, color="#c3c2b7", lw=1)
    ax.axvline(1, color="#c3c2b7", lw=1)
    ax.set_xlabel("tool slowdown under decode (x)")
    ax.set_ylabel("decode slowdown under tool (x)")
    ax.set_title("Interference is two-way and asymmetric")
    ax.legend()
    save(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="+", default=["sleep", "spin", "stream", "pandas"])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--reps", type=int, default=3, help="alone runs per tool")
    ap.add_argument("--warm", type=float, default=10)
    ap.add_argument("--base", type=float, default=6, help="seconds of decode-only baseline")
    a = ap.parse_args()

    out = ROOT / "results/interference" / host()
    out.mkdir(parents=True, exist_ok=True)

    alone = {}
    for name in a.tools:
        t = TOOLS[name]
        ps = [profile(t.argv, cwd=t.cwd) for _ in range(a.reps)]
        alone[name] = {"tool_wall_alone_s": statistics.median(p.wall_s for p in ps),
                       "cores": statistics.median(p.cores for p in ps),
                       "peak_rss_mb": max(p.peak_rss_mb for p in ps)}
        print(name, alone[name], flush=True)

    rows = []
    for b in a.batch:
        with server(b, CTX + 4096, log=out / f"server_B{b}.log") as url:
            rows += asyncio.run(measure(url, b, a.tools, a.warm, a.base))

    cards = pd.DataFrame(rows)
    cards = cards.join(pd.DataFrame(alone).T, on="tool")
    cards["tool_slowdown"] = cards.tool_wall_s / cards.tool_wall_alone_s
    cards["decode_slowdown"] = cards.decode_tps_alone / cards.decode_tps_with_tool
    cards.to_csv(out / "cards.csv", index=False, float_format="%.3f")
    print(cards.round(2).to_string(index=False))
    figure(cards, out / "cards.png")


if __name__ == "__main__":
    main()
