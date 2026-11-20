"""Multi-agent replay runtime.

N agents share one llama-server. Each agent replays recorded TraceLab requests:
for every step it prefills the step's new tokens, decodes exactly the recorded
output tokens, then runs the tool work mapped from the recorded tool call.
A Policy decides when decode chunks and tools may start.

Tools are matched by *work*, not time: a call of recorded duration D runs a
calibrated tool round(D / d0) times, so contention stretches it. Waits that
contention cannot slow (web, human, subagent) are plain sleeps.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

import pandas as pd

from agentsched.llm import Client
from agentsched.tools import category
from agentsched.toolsuite import TOOLS

# ---------------------------------------------------------------- workload


@dataclass
class Step:
    new_tokens: int
    out_tokens: int
    tool: str | None        # suite tool name, or "sleep"
    reps: int = 0           # tool repetitions (or sleep seconds * 1000 for "sleep")


@dataclass
class Request:
    rid: int
    steps: list[Step]


@dataclass
class Shape:
    """How trace requests are rescaled to fit a local model and a time budget."""
    ctx_cap: int = 8192          # compact the context when it would exceed this
    summary_tokens: int = 1024   # context size right after compaction
    system_tokens: int = 2048    # prefill of the first step of a session
    max_new: int = 2048          # cap on tokens appended per step
    max_out: int = 512           # cap on tokens decoded per step
    max_tool_s: float = 30.0     # cap on one recorded tool duration
    time_scale: float = 1.0      # multiply recorded tool durations


# Which suite tool stands for each trace category. "sleep" = pure wait.
TOOLMIX = {
    "synthetic": {"shell": ["spin", "stream", "pandas"], "read": ["spin"], "edit": ["sleep"]},
    "real": {"shell": ["compile", "test", "pandas", "build"], "read": ["grep"], "edit": ["sleep"]},
}


def build_requests(steps: pd.DataFrame, request_ids, tool_s: dict[str, float],
                   shape: Shape, mix: str = "synthetic", seed: int = 0) -> list[Request]:
    """Turn trace requests into replayable scripts. tool_s: calibrated alone time per tool."""
    rng = random.Random(seed)
    table = TOOLMIX[mix]
    out = []
    for rid in request_ids:
        rows = steps[steps.request_id == rid]
        script = []
        for r in rows.itertuples():
            new = min(int(r.new_tokens or 0), shape.max_new)
            outt = max(1, min(int(r.output_tokens or 1), shape.max_out))
            tool, reps = None, 0
            if not r.is_last and r.T_s > 0 and not r.human_wait:
                tools = r.tools if hasattr(r.tools, "__len__") else []
                cat = category(tools[0]) if len(tools) else "shell"
                d = min(r.T_s, shape.max_tool_s) * shape.time_scale
                choices = table.get(cat, ["sleep"])
                tool = rng.choice(choices)
                if tool == "sleep":
                    reps = int(d * 1000)
                else:
                    reps = max(1, round(d / tool_s[tool]))
            script.append(Step(new, outt, tool, reps))
        out.append(Request(int(rid), script))
    return out


def sample_requests(steps: pd.DataFrame, n: int, seed: int = 0, min_tools: int = 1,
                    max_steps: int = 12) -> list[int]:
    g = steps.groupby("request_id").agg(n=("step", "size"), tools=("n_tool_calls", "sum"))
    ok = g[(g.tools >= min_tools) & (g.n <= max_steps)].index.to_list()
    return random.Random(seed).sample(ok, n)


# ---------------------------------------------------------------- policies


class Policy:
    """Scheduling hooks. The default is Naive: never wait."""
    name = "naive"

    def __init__(self):
        self.running: dict[int, str] = {}   # agent -> tool currently running

    async def before_chunk(self, agent: int) -> None:
        pass

    async def before_tool(self, agent: int, tool: str) -> None:
        pass

    def tool_start(self, agent: int, tool: str) -> None:
        self.running[agent] = tool

    def tool_end(self, agent: int, tool: str) -> None:
        self.running.pop(agent, None)


class Stagger(Policy):
    """Pause decoding while any non-sleep tool runs, up to max_wait_s per chunk."""
    name = "stagger"

    def __init__(self, max_wait_s: float = 2.0, heavy: set[str] | None = None):
        super().__init__()
        self.max_wait_s = max_wait_s
        self.heavy = heavy  # None = every tool except sleep
        self.idle = asyncio.Event()
        self.idle.set()

    def _busy(self) -> bool:
        return any(t != "sleep" and (self.heavy is None or t in self.heavy)
                   for t in self.running.values())

    async def before_chunk(self, agent: int) -> None:
        if self._busy():
            try:
                await asyncio.wait_for(self.idle.wait(), self.max_wait_s)
            except asyncio.TimeoutError:
                pass

    def tool_start(self, agent, tool):
        super().tool_start(agent, tool)
        if self._busy():
            self.idle.clear()

    def tool_end(self, agent, tool):
        super().tool_end(agent, tool)
        if not self._busy():
            self.idle.set()


POLICIES = {"naive": Policy, "stagger": Stagger}

# ---------------------------------------------------------------- runtime


@dataclass
class Log:
    t0: float = field(default_factory=time.perf_counter)
    rows: list[dict] = field(default_factory=list)

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def add(self, **kw) -> None:
        self.rows.append(kw)

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


async def run_tool(tool: str, reps: int) -> None:
    if tool == "sleep":
        await asyncio.sleep(reps / 1000)
        return
    t = TOOLS[tool]
    for _ in range(reps):
        p = await asyncio.create_subprocess_exec(*map(str, t.argv), cwd=t.cwd,
                                                 stdout=asyncio.subprocess.DEVNULL,
                                                 stderr=asyncio.subprocess.DEVNULL)
        await p.wait()


class Agent:
    def __init__(self, aid: int, client: Client, policy: Policy, log: Log, shape: Shape,
                 chunk: int | None):
        self.aid, self.c, self.pol, self.log, self.shape = aid, client, policy, log, shape
        self.chunk = chunk
        self.ctx: list[int] = []
        self.rng = random.Random(aid)

    def _fresh(self, n: int) -> list[int]:
        # Arbitrary in-vocabulary ids; content does not matter, only counts.
        return [self.rng.randrange(1000, 30000) for _ in range(n)]

    async def llm_step(self, rid: int, i: int, step: Step) -> None:
        new = self._fresh(self.shape.system_tokens if not self.ctx else step.new_tokens)
        if len(self.ctx) + len(new) + step.out_tokens > self.shape.ctx_cap:
            self.ctx = self._fresh(self.shape.summary_tokens)   # compaction
            self.log.add(agent=self.aid, rid=rid, step=i, kind="compact", t0=self.log.now(),
                         t1=self.log.now())
        self.ctx += new
        left, first = step.out_tokens, True
        while left > 0:
            n = left if self.chunk is None else min(self.chunk, left)
            w0 = self.log.now()
            await self.pol.before_chunk(self.aid)
            t0 = self.log.now()
            if t0 - w0 > 1e-3:
                self.log.add(agent=self.aid, rid=rid, step=i, kind="wait", t0=w0, t1=t0)
            r = await self.c.complete(self.ctx, n, slot=self.aid)
            self.log.add(agent=self.aid, rid=rid, step=i, kind="prefill" if first else "decode",
                         t0=t0, t1=self.log.now(), n_prefill=r["n_prefill"],
                         n_decode=r["n_decode"], prefill_ms=r["prefill_ms"],
                         decode_ms=r["decode_ms"])
            self.ctx += r["tokens"] or self._fresh(n)
            left -= n
            first = False

    async def tool_step(self, rid: int, i: int, step: Step) -> None:
        tool = step.tool
        assert tool is not None
        w0 = self.log.now()
        await self.pol.before_tool(self.aid, tool)
        t0 = self.log.now()
        if t0 - w0 > 1e-3:
            self.log.add(agent=self.aid, rid=rid, step=i, kind="wait", t0=w0, t1=t0)
        self.pol.tool_start(self.aid, tool)
        try:
            await run_tool(tool, step.reps)
        finally:
            self.pol.tool_end(self.aid, tool)
        self.log.add(agent=self.aid, rid=rid, step=i, kind="tool", tool=tool,
                     reps=step.reps, t0=t0, t1=self.log.now())

    async def run(self, requests: list[Request]) -> None:
        for req in requests:
            t0 = self.log.now()
            for i, s in enumerate(req.steps):
                await self.llm_step(req.rid, i, s)
                if s.tool:
                    await self.tool_step(req.rid, i, s)
            self.log.add(agent=self.aid, rid=req.rid, step=-1, kind="request", t0=t0,
                         t1=self.log.now())


async def replay(url: str, per_agent: list[list[Request]], policy: Policy, shape: Shape,
                 chunk: int | None = 32) -> pd.DataFrame:
    """Run one agent per entry of per_agent; returns the event log."""
    log = Log()
    client = Client(url)
    agents = [Agent(a, client, policy, log, shape, chunk) for a in range(len(per_agent))]
    try:
        await asyncio.gather(*(ag.run(reqs) for ag, reqs in zip(agents, per_agent)))
    finally:
        await client.close()
    return log.df()


def tokens_match(solo: pd.DataFrame, log: pd.DataFrame) -> bool:
    """Correctness: every step decodes the same number of tokens as in the solo run."""
    def per_step(d):
        d = d[d.kind.isin(["prefill", "decode"])]
        return d.groupby(["rid", "step"]).n_decode.sum()
    s, g = per_step(solo), per_step(log)
    return bool(g.index.isin(s.index).all() and (g == s.loc[g.index]).all())


def summarize(log: pd.DataFrame) -> dict:
    req = log[log.kind == "request"]
    act = req.t1 - req.t0
    makespan = log.t1.max()
    llm = log[log.kind.isin(["prefill", "decode"])]
    return {
        "agents": log.agent.nunique(),
        "requests": len(req),
        "makespan_s": makespan,
        "throughput_rpm": len(req) / makespan * 60,
        "active_p50_s": act.median(),
        "active_p95_s": act.quantile(0.95),
        "decode_tps": llm.n_decode.sum() / (llm.decode_ms.sum() / 1000) if len(llm) else 0,
        "tool_s": (log[log.kind == "tool"].eval("t1 - t0")).sum(),
        "wait_s": (log[log.kind == "wait"].eval("t1 - t0")).sum(),
    }
