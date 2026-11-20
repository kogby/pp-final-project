"""llama-server launcher and async client.

Agents talk to one llama-server with N slots (continuous batching). Prompts are
token-id lists so prefill lengths match the trace exactly; `cache_prompt` plus a
fixed slot per agent keeps each agent's prefix in KV cache between steps.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from agentsched.traces import ROOT

MODEL = ROOT / "data/models/Qwen3-8B-Q4_K_M.gguf"


@contextmanager
def server(n_slots: int, ctx_per_slot: int, model: Path = MODEL, port: int = 8080,
           log: Path | None = None):
    """Run llama-server for the duration of the block."""
    argv = ["llama-server", "-m", str(model), "--port", str(port), "-np", str(n_slots),
            "-c", str(n_slots * ctx_per_slot), "-fa", "on", "--no-webui",
            "--cont-batching", "--metrics", "--slots"]
    out = open(log, "w") if log else subprocess.DEVNULL
    proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({proc.returncode}); see {log}")
            time.sleep(0.5)
        else:
            raise TimeoutError("llama-server did not become healthy")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log:
            out.close()


class Client:
    def __init__(self, url: str):
        self.url = url
        self.http = httpx.AsyncClient(base_url=url, timeout=None)

    async def complete(self, tokens: list[int], n_predict: int, slot: int) -> dict:
        """Prefill `tokens` (reusing cached prefix) and decode exactly n_predict tokens."""
        r = await self.http.post("/completion", json={
            "prompt": tokens,
            "n_predict": n_predict,
            "ignore_eos": True,
            "temperature": 0.0,
            "seed": 0,
            "cache_prompt": True,
            "id_slot": slot,
            "return_tokens": True,
            "samplers": ["top_k"],
            "top_k": 1,
        })
        r.raise_for_status()
        j = r.json()
        t = j.get("timings", {})
        return {
            "tokens": j.get("tokens", []),
            "n_cached": t.get("cache_n", 0),
            "n_prefill": t.get("prompt_n", 0),
            "prefill_ms": t.get("prompt_ms", 0.0),
            "n_decode": t.get("predicted_n", 0),
            "decode_ms": t.get("predicted_ms", 0.0),
        }

    async def close(self) -> None:
        await self.http.aclose()


async def _smoke(url: str) -> None:
    c = Client(url)
    ctx = list(range(1000, 1512))
    r = await c.complete(ctx, 32, slot=0)
    print({k: v for k, v in r.items() if k != "tokens"})
    r = await c.complete(ctx + r["tokens"] + list(range(2000, 2128)), 32, slot=0)
    print({k: v for k, v in r.items() if k != "tokens"}, "(second step should reuse the cache)")
    await c.close()


if __name__ == "__main__":
    with server(2, 4096, log=ROOT / "data/llama-server.log") as u:
        asyncio.run(_smoke(u))
