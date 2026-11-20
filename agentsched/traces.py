"""Load TraceLab into a per-step table: one row per LLM round.

A *request* is the span from one user message to the next. Inside a request,
step i has
    L_s  LLM time: round input arrives -> last event the model emits
    T_s  tool time: last emitted event -> next round's input arrives
The last step of a request has T_s = 0 (the next input is a human).

T_s is wall time seen by the agent, so it includes harness overhead and, for
Claude Code, any time a human spent approving a tool.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from agentsched.tools import CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/tracelab/syfi_coding_trace.duckdb"
CACHE = ROOT / "data/tracelab/steps.parquet"

INPUT_EVENTS = "('user_message', 'tool_result')"
OUTPUT_EVENTS = "('reasoning', 'text', 'tool_call')"

STEPS_SQL = f"""
with ev as (
  select round_pk,
         min(timestamp) filter (where event_type in {INPUT_EVENTS})  as t_in,
         max(timestamp) filter (where event_type in {OUTPUT_EVENTS}) as t_out,
         min(timestamp) as t_first,
         count(*) filter (where event_type = 'tool_call') as n_tool_calls
  from timing_events group by round_pk
),
tc as (
  select round_pk, list(tool_name order by tool_index) as tools,
         sum(result_chars) as tool_result_chars
  from tool_calls group by round_pk
)
select r.round_pk, r.provider, r.model, r.session_id, r.round_index,
       r.current_user_message_count > 0 as is_user_turn,
       coalesce(ev.t_in, ev.t_first) as t_in, ev.t_out, ev.n_tool_calls,
       r.input_tokens_total as ctx_tokens, r.newly_append_tokens as new_tokens,
       r.output_tokens, tc.tools, tc.tool_result_chars
from rounds r join ev using (round_pk) left join tc using (round_pk)
order by r.session_id, r.round_index
"""


def build_steps(db: Path = DB) -> pd.DataFrame:
    con = duckdb.connect(str(db), read_only=True)
    df = con.execute(STEPS_SQL).df()

    # A request starts at the first round of a session or at any user turn.
    first = df.session_id.ne(df.session_id.shift())
    df["request_id"] = (first | df.is_user_turn).cumsum() - 1
    df["step"] = df.groupby("request_id").cumcount()

    df["L_s"] = (df.t_out - df.t_in).dt.total_seconds().clip(lower=0).fillna(0)
    nxt_same_req = df.request_id.eq(df.request_id.shift(-1))
    gap = (df.t_in.shift(-1) - df.t_out).dt.total_seconds().clip(lower=0)
    df["T_s"] = gap.where(nxt_same_req, 0.0).fillna(0)
    df["is_last"] = ~nxt_same_req
    human = CATEGORIES["human"]
    df["human_wait"] = df.tools.map(lambda t: hasattr(t, "__len__") and any(x in human for x in t))
    return df


def load_steps(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    df = build_steps()
    df.to_parquet(CACHE, index=False)
    return df


def tool_calls(db: Path = DB) -> pd.DataFrame:
    con = duckdb.connect(str(db), read_only=True)
    return con.execute(
        "select t.*, r.provider from tool_calls t join rounds r using (round_pk)"
    ).df()
