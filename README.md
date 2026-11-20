# Scheduling Concurrent Coding Agents on Unified-Memory Hardware

15-418/618 Fall 2026 final project. Jerry Chen, Ethan Chiu.
Project page: https://kogby.github.io/pp-final-project/ · Plan: [ROADMAP.md](ROADMAP.md)

## Layout

```
agentsched/    library: trace loading, headroom, tools, plotting (later: replay, schedulers)
experiments/   one script per study; each writes results/<study>/
results/       CSVs and figures (committed)
data/          traces and model weights (gitignored)
docs/          project website (GitHub Pages)
```

## Setup

```bash
uv sync --extra dev
brew install llama.cpp                     # or build with CUDA on the GB10
gh release download v0.0.1 -R uw-syfi/TraceLab -D data/tracelab -p '*.duckdb'
uvx --from huggingface_hub hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir data/models
```

## Run

```bash
uv run experiments/workload.py                      # trace workload model + headroom
caffeinate -i uv run experiments/llm_rates.py       # prefill/decode/batched rates
caffeinate -i uv run experiments/interference.py    # tool data cards
cd experiments && caffeinate -i uv run scaling.py --n 1 2 4 8   # Solo/Naive/Stagger/Ideal sweep
uv run experiments/timeline.py results/scaling/<host>/log_naive_N4.csv
uv run pytest -q tests
```

Results are written per host (`results/<study>/<host>/`). On laptops, run plugged in under
`caffeinate`, and let the machine cool between sweeps.

## Deadlines

| Item | Due |
|---|---|
| Milestone report | Mon Nov 23 |
| Final report | Thu Dec 10 |
| Poster session | Fri Dec 11 |
