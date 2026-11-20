"""Real and synthetic tools that stand in for trace tool calls.

Trace inputs are stripped, so each recorded call is mapped to one of these by
category and duration. Every tool is an unmodified program run as a subprocess.

    python -m agentsched.toolsuite setup        # fetch workspace, build kernels
    python -m agentsched.toolsuite run NAME     # run one tool with profiling
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agentsched.traces import ROOT

WS = ROOT / "data/workspace"
KERNELS = Path(__file__).parent / "kernels"
BIN = WS / "bin"
NCPU = os.cpu_count() or 4
PY = sys.executable

LLAMA = WS / "llama.cpp"
PYREPO = WS / "more-itertools"


@dataclass(frozen=True)
class Tool:
    name: str
    category: str      # trace category it stands for
    argv: tuple[str, ...]
    cwd: Path = WS
    reset: tuple[str, ...] = ()  # run before each call to make it repeatable

    def run(self, **kw) -> subprocess.CompletedProcess:
        if self.reset:
            subprocess.run(self.reset, cwd=self.cwd, capture_output=True)
        return subprocess.run(self.argv, cwd=self.cwd, capture_output=True, text=True, **kw)


def _tools() -> dict[str, Tool]:
    inc = ("-Iinclude", "-Iggml/include", "-Isrc")
    tools = [
        Tool("sleep", "synthetic", ("sleep", "5")),
        Tool("spin", "synthetic", (PY, "-c", "import time\nt=time.time()\nwhile time.time()-t<5: pass")),
        Tool("stream", "synthetic", (str(BIN / "stream"), str(NCPU), "256", "5")),
        Tool("grep", "read", ("rg", "-c", "--no-messages", "ggml_tensor|llama_", "."), LLAMA),
        Tool("compile", "shell", ("clang++", "-std=c++17", "-O2", *inc, "-c",
                                  "src/llama-model.cpp", "-o", "/dev/null"), LLAMA),
        Tool("build", "shell", ("cmake", "--build", "build", "-j", str(NCPU), "--target", "llama"),
             LLAMA, reset=("cmake", "--build", "build", "--target", "clean")),
        Tool("test", "shell", (PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"), PYREPO),
        Tool("pandas", "shell", (PY, str(KERNELS / "pandas_job.py"), "25")),
        Tool("pandas-small", "shell", (PY, str(KERNELS / "pandas_job.py"), "8")),
    ]
    return {t.name: t for t in tools}


TOOLS = _tools()


def setup() -> None:
    WS.mkdir(parents=True, exist_ok=True)
    BIN.mkdir(exist_ok=True)
    subprocess.run(["clang", "-O2", "-pthread", str(KERNELS / "stream.c"), "-o", str(BIN / "stream")],
                   check=True)
    if not LLAMA.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp",
                        str(LLAMA)], check=True)
    if not (LLAMA / "build").exists():
        subprocess.run(["cmake", "-B", "build", "-DGGML_METAL=OFF", "-DLLAMA_CURL=OFF",
                        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                        "-DLLAMA_BUILD_SERVER=OFF", "-DCMAKE_BUILD_TYPE=Release"],
                       cwd=LLAMA, check=True, capture_output=True)
    if not PYREPO.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/more-itertools/more-itertools", str(PYREPO)], check=True)
    missing = [t for t in ("rg", "clang++", "cmake") if shutil.which(t) is None]
    if missing:
        print("missing on PATH:", missing)


if __name__ == "__main__":
    from agentsched.profile import profile

    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "setup":
        setup()
    elif cmd == "run":
        t = TOOLS[sys.argv[2]]
        if t.reset:
            subprocess.run(t.reset, cwd=t.cwd, capture_output=True)
        print(profile(t.argv, cwd=t.cwd))
    else:
        for t in TOOLS.values():
            print(f"{t.name:8s} {t.category:9s} {' '.join(map(str, t.argv))}")
