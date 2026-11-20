"""Profile a command: wall time, CPU time, average cores, peak RSS.

Samples the whole process tree with psutil, so multi-process tools (cmake,
pytest-xdist) are counted. Memory-bandwidth counters need root on macOS
(powermetrics) and perf on Linux; until then bandwidth pressure is measured
indirectly by the slowdown of a co-running probe (see experiments/interference.py).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


@dataclass
class Profile:
    wall_s: float
    cpu_s: float
    cores: float        # cpu_s / wall_s
    peak_rss_mb: float
    returncode: int

    def as_dict(self) -> dict:
        return asdict(self)


def host() -> str:
    """Short machine tag used to separate results from different hardware."""
    import platform
    chip = ""
    if platform.system() == "Darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    return (chip or platform.node()).replace(" ", "-")


def _tree(p: psutil.Process) -> list[psutil.Process]:
    try:
        return [p, *p.children(recursive=True)]
    except psutil.Error:
        return [p]


def profile(argv, cwd: Path | None = None, interval: float = 0.05) -> Profile:
    t0 = time.perf_counter()
    proc = subprocess.Popen(list(map(str, argv)), cwd=cwd,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    root = psutil.Process(proc.pid)
    cpu: dict[int, float] = {}
    peak = 0.0
    while proc.poll() is None:
        rss = 0.0
        for p in _tree(root):
            try:
                with p.oneshot():
                    t = p.cpu_times()
                    cpu[p.pid] = t.user + t.system
                    rss += p.memory_info().rss
            except psutil.Error:
                pass
        peak = max(peak, rss)
        time.sleep(interval)
    wall = time.perf_counter() - t0
    cpu_s = sum(cpu.values())
    return Profile(wall, cpu_s, cpu_s / wall if wall else 0.0, peak / 2**20, proc.returncode)
