"""Single-agent overlap headroom.

For a request with steps (L_i, T_i):
    one-step   H1 = sum_i min(L_i, T_i)              / sum_i (L_i + T_i)
    unlimited  Hu = min(sum_i L_i, sum_i T_i)        / sum_i (L_i + T_i)
Speedup is bounded by 1 / (1 - H). "Unlimited" assumes perfect speculation
with no lookahead limit, so the request can be no shorter than its longer lane.

Corpus numbers are time-weighted: sums over all requests, not a mean of ratios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LocalLLM:
    """Step latency for a local model: prefill new tokens, then decode output.

    Rates are tokens/s, measured with llama-bench at the given context size.
    Context beyond `ctx` is truncated, which is what a local agent would do.
    """

    name: str
    ctx: int
    prefill_tps: float
    decode_tps: float

    def step_time(self, new_tokens: pd.Series, output_tokens: pd.Series) -> pd.Series:
        new = new_tokens.fillna(0).clip(upper=self.ctx)
        out = output_tokens.fillna(0)
        return new / self.prefill_tps + out / self.decode_tps


def capped(steps: pd.DataFrame, cap_s: float | None) -> pd.DataFrame:
    """Drop T for steps waiting on a human; clip L and T at cap_s (longer gaps
    are idle time such as a sleeping laptop or unattended approval prompt)."""
    df = steps.copy()
    df.loc[df.human_wait, "T_s"] = 0.0
    if cap_s is not None:
        df["L_s"] = df.L_s.clip(upper=cap_s)
        df["T_s"] = df.T_s.clip(upper=cap_s)
    return df


def headroom(steps: pd.DataFrame, L: pd.Series | None = None) -> dict:
    """Corpus headroom. Pass `L` to replace recorded LLM time with a model."""
    L = steps.L_s if L is None else L
    T = steps.T_s
    req = steps.request_id
    total = L.sum() + T.sum()
    one = np.minimum(L, T).sum()
    per_req = pd.DataFrame({"llm": L, "tool": T, "r": req}).groupby("r").sum()
    unl = np.minimum(per_req.llm, per_req.tool).sum()
    h1, hu = one / total, unl / total
    return {
        "requests": len(per_req),
        "tool_share": T.sum() / total,
        "H_one": h1,
        "bound_one": 1 / (1 - h1),
        "H_unl": hu,
        "bound_unl": 1 / (1 - hu),
    }
