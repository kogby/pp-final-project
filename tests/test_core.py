import numpy as np
import pandas as pd
import pytest

from agentsched.headroom import capped, headroom
from agentsched.replay import Shape, build_requests


def steps(L, T, req=None, **extra):
    n = len(L)
    return pd.DataFrame({
        "request_id": req or [0] * n, "step": range(n), "L_s": L, "T_s": T,
        "human_wait": [False] * n, "is_last": [i == n - 1 for i in range(n)], **extra,
    })


def test_headroom_one_step_and_unlimited():
    # L=[4,1], T=[1,4]: one-step overlap hides min(4,1)+min(1,4)=2 of 10.
    h = headroom(steps([4, 1], [1, 4]))
    assert h["H_one"] == pytest.approx(0.2)
    # Unlimited: request no shorter than max(sum L, sum T)=5 -> hides 5 of 10.
    assert h["H_unl"] == pytest.approx(0.5)
    assert h["bound_unl"] == pytest.approx(2.0)


def test_cap_and_human_wait():
    df = steps([1, 1], [100, 0])
    df.loc[0, "human_wait"] = True
    assert capped(df, None).T_s.sum() == 0
    df.loc[0, "human_wait"] = False
    assert capped(df, 10).T_s.max() == 10


def test_build_requests_work_matched():
    df = steps([1, 1, 1], [9.0, 2.0, 0.0], new_tokens=[100, 5000, 10],
               output_tokens=[50, 900, 0], tools=[np.array(["Bash"]), np.array(["Edit"]), None])
    shape = Shape(max_new=2048, max_out=512, max_tool_s=30)
    (req,) = build_requests(df, [0], {"spin": 3.0, "stream": 3.0, "pandas": 3.0}, shape)
    a, b, c = req.steps
    assert a.reps == 3 and a.tool in {"spin", "stream", "pandas"}   # 9 s / 3 s per run
    assert b.tool == "sleep" and b.reps == 2000                      # edit -> 2 s wait
    assert b.new_tokens == 2048 and b.out_tokens == 512              # capped
    assert c.tool is None and c.out_tokens == 1                      # last step: no tool
