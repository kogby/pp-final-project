"""A pandas script typical of agent data tasks: build, join, group, sort.

    python pandas_job.py <million rows>
"""

import sys
import time

import numpy as np
import pandas as pd

n = int(float(sys.argv[1]) * 1_000_000) if len(sys.argv) > 1 else 5_000_000
t0 = time.perf_counter()
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "user": rng.integers(0, n // 50, n),
    "item": rng.integers(0, 10_000, n),
    "price": rng.random(n) * 100,
    "qty": rng.integers(1, 10, n),
})
items = pd.DataFrame({"item": np.arange(10_000), "cat": rng.integers(0, 50, 10_000)})
df = df.merge(items, on="item")
df["rev"] = df.price * df.qty
by_user = df.groupby("user").agg(rev=("rev", "sum"), n=("qty", "size"))
by_cat = df.groupby("cat").rev.describe()
top = by_user.sort_values("rev", ascending=False).head(10)
print(f"rows={n} users={len(by_user)} top={top.rev.iloc[0]:.1f} cats={len(by_cat)} "
      f"t={time.perf_counter() - t0:.2f}s")
