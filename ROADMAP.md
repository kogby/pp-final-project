# Roadmap

Status as of Nov 20. Milestone report due **Nov 23**, final report **Dec 10**, poster **Dec 11**.
Check items off as they land; each item names the script that produces its result.

## Phase 1 — Workload model (Nov 9–20)

- [x] Load TraceLab v0.0.1 into a per-step table (`agentsched/traces.py`)
- [x] Request shape, tool categories, latency tail (`experiments/workload.py`)
- [x] Single-agent headroom vs idle cap, cloud LLM times
- [ ] Headroom with local LLM times (needs Phase 2 rates)
- [ ] Reconcile tool share with the proposal table (32% there vs ~50% here; definition differs)

Findings so far (30 min idle cap): 41.5k requests, 8.6 steps and 10.4 tool calls per
request, 38% call no tool. 3.1% of calls exceed 60 s and hold 84% of tool time.
One-step headroom bound 1.06x, unlimited 1.20x. Median end-of-request context is
147k tokens, well beyond what a local 8B model holds, so replay must rescale context.

## Phase 2 — LLM rates (Nov 20–22)

- [x] `llama-bench` prefill/decode rate vs context and batch size (`experiments/llm_rates.py`) — M4
- [x] Fit `LocalLLM` per (model, ctx) → `results/llm_rates/rates.csv`; headroom rows added
- [x] Batched decode step time vs N × context (`llama-batched-bench`) — M4, N ≤ 4
- [ ] Repeat on M5 Pro and GB10 (full sweep: `uv run experiments/llm_rates.py`)

M4 findings (Qwen3-8B Q4_K_M): decode 19.1 → 11.2 tok/s and prefill 195 → 76 tok/s from 0 → 16k
context. Batching 4 agents only doubles aggregate decode (18 → 36.5 tok/s at 1k ctx; 11.1 → 21.9
at 8k). With local LLM times the single-agent bound is 1.05–1.07x one-step, 1.11–1.17x unlimited.

## Phase 3 — Tool data cards (Nov 21–26)

- [x] Tool suite (`agentsched/toolsuite.py`): `rg`, `clang` build, `pytest`, `pandas`, and
      synthetic tools (sleep, CPU loop, bandwidth kernel)
- [x] Profiler (`agentsched/profile.py`): CPU %, RSS, wall time, sampled during a run
- [x] Interference matrix: tool slowdown under decode, decode slowdown under tool
      (`experiments/interference.py`; measurements paired in time to cancel drift)
- [ ] Rerun on a machine with active cooling (M4 Air is fanless and throttles; see below)

M4 data cards (preliminary): interference is two-way and asymmetric. `stream` (bandwidth hog)
slows decode 2.1–2.3x but is itself unaffected; `pandas` (random access, 1 core) slows 2.6–2.9x
under decode (2x even at 0.85 GB, so not just capacity) while slowing decode only ~1.2x.
`spin` (pure CPU) and `sleep` do not interact. A scheduler must treat these differently.
- [x] Map trace calls to suite tools (`TOOLMIX` in `agentsched/replay.py`); work-matched:
      a call of recorded duration D runs round(D / d0) calibrated repetitions
- [ ] Real-program tools (`grep`, `compile`, `build`, `test`) need `toolsuite setup`, which
      clones llama.cpp and more-itertools into `data/workspace/` (not yet run)

Bandwidth note (M4, 120 GB/s peak): one CPU thread running a STREAM triad reaches 78 GB/s,
ten threads 91 GB/s. A single CPU tool can take most of the bandwidth decode needs.

## Phase 4 — Replay runtime (Nov 22–27)

- [x] `agentsched/replay.py`: asyncio, N sessions, each step = `llama-server` request with
      the trace's new/output token counts (`ignore_eos`, `n_predict`), then the mapped tool
- [x] Event log per agent (step start/end, tool start/end) → timeline plot (`experiments/timeline.py`)
- [x] Correctness: decoded tokens per step identical alone vs in a group (checked every run,
      `tokens_match_solo` in summary.csv); smoke test passes at N=1,2 for Naive and Stagger
- [x] **Ideal**: one Solo pass gives each request's isolated time (`experiments/scaling.py`)

## Phase 5 — Scaling study (Nov 26–Dec 2)

- [ ] Sweep N for **Solo**, **Naive**, **Stagger** on M-series and GB10 (`experiments/scaling.py`)
- [ ] Throughput, p95 active time, scaling efficiency, recovered gap
- [ ] Identify the limit per machine: KV capacity vs bandwidth

## Phase 6 — Tool-aware scheduler (Nov 30–Dec 5)

- [ ] Policy using data cards: pause or throttle decode only for heavy tools; admission limit;
      core placement on GB10
- [ ] Compare against Naive, Stagger, Ideal. Targets: recover ≥50% of the Naive–Ideal gap in
      p95 active time at N≥8 on GB10; match Stagger p95 with ≥20% more throughput

## Stretch

- [ ] Speculation in batch slack (canonicalized outcomes, isolated workspace copies)
- [ ] Online policy sensing contention from decode latency

## Open questions

- Measurement hygiene on laptops: run under `caffeinate -i` (a smoke run was invalidated by
  sleep), plug in, close other apps, and let the machine cool between sweeps. The M4 Air
  dropped from 19 to ~8 tok/s decode after ~40 min of load.

- This laptop is an M4 / 16 GB, not the M5 Pro / 24 GB in the proposal. Run M-series numbers
  on whichever machine the final report will cite.
