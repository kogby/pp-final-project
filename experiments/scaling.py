"""Sweep the number of concurrent agents N under each policy.

    uv run experiments/scaling.py --n 1 2 4 --policies naive stagger --reqs 2

Every agent a replays the same request list at every N and policy, so runs are
comparable. "Ideal" comes from one Solo pass over all agents' requests: each
request's isolated active time, with throughput N x solo throughput.
Writes results/scaling/<host>/{summary.csv, log_<policy>_N<n>.csv}.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict

import pandas as pd

from agentsched.llm import server
from agentsched.profile import host, profile
from agentsched.replay import POLICIES, TOOLMIX, Policy, Shape, build_requests, replay, \
    sample_requests, summarize
from agentsched.toolsuite import TOOLS
from agentsched.traces import ROOT, load_steps


def calibrate(names: set[str], reps: int = 3) -> dict[str, float]:
    out = {}
    for n in sorted(names - {"sleep"}):
        t = TOOLS[n]
        out[n] = statistics.median(profile(t.argv, cwd=t.cwd).wall_s for _ in range(reps))
        print(f"calibrated {n}: {out[n]:.2f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--policies", nargs="+", default=["naive", "stagger"])
    ap.add_argument("--reqs", type=int, default=2, help="requests per agent")
    ap.add_argument("--mix", default="synthetic", choices=list(TOOLMIX))
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-tool-s", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ideal", action="store_true")
    a = ap.parse_args()

    out = ROOT / "results/scaling" / host()
    out.mkdir(parents=True, exist_ok=True)
    shape = Shape(ctx_cap=a.ctx, max_tool_s=a.max_tool_s)
    nmax = max(a.n)

    steps = load_steps()
    rids = sample_requests(steps, nmax * a.reqs, seed=a.seed)
    names = {t for v in TOOLMIX[a.mix].values() for t in v}
    tool_s = calibrate(names)
    reqs = build_requests(steps, rids, tool_s, shape, a.mix, a.seed)
    per_agent = [reqs[i * a.reqs:(i + 1) * a.reqs] for i in range(nmax)]
    (out / "config.json").write_text(json.dumps(
        {"args": vars(a), "shape": asdict(shape), "tool_s": tool_s, "request_ids": rids}, indent=1))

    rows = []
    if not a.no_ideal:
        with server(1, a.ctx, log=out / "server_solo.log") as url:
            # One agent runs every request back to back: isolated per-request times.
            log = asyncio.run(replay(url, [reqs], Policy(), shape, a.chunk))
        log.to_csv(out / "log_solo.csv", index=False)
        solo = log[log.kind == "request"].assign(active=lambda d: d.t1 - d.t0)
        for n in a.n:
            act = solo[solo.rid.isin([r.rid for agent in per_agent[:n] for r in agent])].active
            rows.append({"policy": "ideal", "agents": n, "requests": len(act),
                         "throughput_rpm": n * 60 / solo.active.mean(),
                         "active_p50_s": act.median(), "active_p95_s": act.quantile(0.95)})

    for n in a.n:
        for p in a.policies:
            with server(n, a.ctx, log=out / f"server_{p}_N{n}.log") as url:
                log = asyncio.run(replay(url, per_agent[:n], POLICIES[p](), shape, a.chunk))
            log.to_csv(out / f"log_{p}_N{n}.csv", index=False)
            s = {"policy": p, **summarize(log)}
            print(s, flush=True)
            rows.append(s)
            pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
