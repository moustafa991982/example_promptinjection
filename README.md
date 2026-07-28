# pi-harness

A red-team harness for measuring how well a large language model resists **prompt-injection attacks**. It runs a corpus of known injection payloads against a target model, scores whether each attack succeeds, and reports an **Attack Success Rate (ASR)** you can track across models, configurations, and system prompts.

> **Responsible use.** This is a defensive robustness-testing tool. It is intended for evaluating models and deployments you own or are authorized to assess. Use it to *harden* systems — measure a baseline, add mitigations, and confirm the ASR drops.

---

## What it does

- Loads a prompt-injection corpus (the Giskard set, cloned automatically on first run) plus any custom injections you supply.
- Sends each attack to a target model over an OpenAI-compatible or Ollama backend.
- Scores every response with a string matcher and an independent **canary** check.
- Aggregates results into ASR, with errored trials excluded from the denominator, and writes machine- and human-readable outputs for further analysis.

It supports two threat models via the `--vector` flag:

- **`direct`** — the payload goes in the user turn. Measures base-model alignment.
- **`indirect`** — the payload is wrapped in an untrusted "retrieved document" behind an innocuous user turn. Measures deployed **RAG** risk.
- **`both`** — runs each prompt under both vectors.

---

## Installation

```bash
./setup.sh
# or, manually:
pip install -r requirements.txt
```

Requires a reachable target model — either an OpenAI-compatible endpoint or a local [Ollama](https://ollama.com) server.

---

## Quick start

Run against a local Ollama model and write a baseline:

```bash
python -m piharness.cli \
  --backend ollama \
  --base-url http://localhost:11434 \
  --model llama3.1:latest \
  --timeout 600 \
  --outdir runs/baseline \
  run --repeats 3 --concurrency 1
```

Preview the trial plan without making any network calls:

```bash
python -m piharness.cli run --dry-run
```

Measure your **actual deployment** rather than the base model — apply a system profile, test both vectors, and append your own injections:

```bash
python -m piharness.cli \
  --backend ollama --model llama3.1:latest --outdir runs/deployed \
  run --system-profile vehicle_support --vector both \
      --extra examples/custom_injections.json
```

---

## Key parameters

Resolution order for every setting is **CLI flag → YAML value → built-in default**. Anything below can live in `configs/default.yaml` and be overridden per run.

### Target and transport (global)

| Flag | Default | Notes |
|---|---|---|
| `--config` | `configs/default.yaml` | Path to the YAML config. |
| `--base-url` | `http://localhost:11434/v1` | Server root. The `openai` backend needs `/v1`; the `ollama` backend must not have it. |
| `--model` | `llama3.1:8b` | Passed through verbatim. Ollama matches tags literally (`:latest` ≠ `:8b`). |
| `--backend` | `openai` | Switches endpoint path, payload shape, auth header, and response parsing together. |
| `--temperature` | `0.0` | **Affects results.** Greedy at `0.0` — repeats measure no real variance. Use `0.7–1.0` with `--repeats`; use `0.0` for a reproducible regression baseline. |
| `--max-tokens` | `512` | **Affects results.** Too low truncates before the marker appears (false negatives). Watch for `finish_reason = length`. |
| `--timeout` | `180` | Read timeout (seconds). Retries with backoff, then records an error; errored trials are excluded from ASR. |
| `--seed` | *(none)* | Base seed; repeat *i* uses `seed + i`. Redundant at temperature `0.0`. |

### Corpus and output (global)

| Flag | Default | Notes |
|---|---|---|
| `--dataset-dir` | `data/prompt-injections` | Where the corpus lives; cloned on first use if absent. |
| `--no-clone` | off | Error instead of cloning — pin a corpus revision for a stable assessment. |
| `--outdir` | `runs/latest` | Destination for all six output files, and the **resume key**: same directory resumes, new directory starts fresh. One directory per configuration you compare. |

### Case selection (`run`)

| Flag | Notes |
|---|---|
| `--groups` | Filter by family (e.g. `DAN`, `DUDE`) or coarse group (`Jailbreak`, `Hijacking`). Case-insensitive. |
| `--languages` | Filter on the corpus language column. |
| `--extra` | JSON file of your own injections, appended before filtering. |
| `--limit` | Take the first *N* cases after filtering — a head slice for smoke tests, not a random sample. |

### Execution shape (`run`)

| Flag | Default | Notes |
|---|---|---|
| `--repeats` | `1` | **Affects results.** Trials per case per vector. Aggregation is **max-over-repeats** — a prompt counts vulnerable if *any* repeat lands. |
| `--concurrency` | `4` | **Affects results.** Set to `1` for Ollama, which serializes requests by default; higher only queues them and risks timeouts. |
| `--vector` | `direct` | **Affects results.** `direct` / `indirect` / `both` (see threat models above). |

### Deployment context (`run`)

| Flag | Default | Notes |
|---|---|---|
| `--system-profile` | `none` | **Affects results.** Names a key under `system_profiles` in the YAML. `none` sends no system message. |
| `--system-prompt-file` | *(none)* | **Affects results.** Reads a system prompt from a file; takes precedence over `--system-profile`. Test your real deployed prompt this way. |
| `--canary` | *(YAML)* | **Affects results.** Substituted into `{{CANARY}}` and checked verbatim in every response, independent of the string matcher. |

### Resume and preview (`run`)

| Flag | Notes |
|---|---|
| `--no-resume` | Skips the completed-trial check. The JSONL is append-mode, so this **appends duplicates** — point `--outdir` somewhere new instead. |
| `--dry-run` | Prints the trial plan and sample prompts, then exits before any network call. |

### Reporting (`report`)

| Flag | Notes |
|---|---|
| `--results` | Read a specific JSONL file and rescore — used to re-run scoring after editing `matcher.py`, with no re-querying of the model. |

### YAML-only settings

`num_ctx` (Ollama context window — too small silently truncates long attacks), `keep_alive`, `max_retries`, `top_p`, `api_key`, and `extra_body` are set in the config file rather than on the command line.

> **The three settings that change your numbers:** `--temperature` (repeats measure nothing at `0.0`), `num_ctx` (silent prompt truncation), and `--system-profile` (base-model alignment vs. your real deployment). Everything else affects ergonomics or runtime.

---

## Project structure

```
configs/     YAML configuration (default.yaml, system profiles)
data/        prompt-injection corpora (auto-cloned on first run)
examples/    sample configs and custom injection sets
runs/        execution outputs — one directory per configuration (gitignored)
tests/       unit / integration tests
setup.sh     environment setup
```

---

## Output

Each run writes six files to `--outdir`, including:

- **`results.jsonl`** — one record per trial (re-scorable via `report --results`).
- **`results.csv`** — flat table for spreadsheet or notebook analysis.
- **`run_meta.json`** — the exact configuration the run executed under, for reproducibility.

ASR is computed over completed trials; errored trials are excluded from the denominator.

---

## Results

Example assessment of **`llama3.1:latest`** via the Ollama backend, `direct` vector, no system profile (base-model alignment). Two runs are included: a small **smoke** run that confirms the harness works end to end, and a **baseline** run used for the actual measurement.

The headline metric is **case-level ASR** using the harness's own aggregation — *max-over-repeats*: a case counts vulnerable if **any** repeat lands, because an attacker only needs one. Trial-level ASR (every repeat counted separately) is shown alongside for transparency.

<img width="1780" height="1019" alt="asr_by_family" src="https://github.com/user-attachments/assets/1ebee083-f326-43df-94a6-aac22bc08708" />

### Baseline

35 cases · 154 trials · direct vector · 3 repeats per case (10 for a subset) · 0 errors · 0 canary leaks.

| Metric | Value |
|---|---:|
| Cases assessed | 35 |
| Cases vulnerable (any repeat landed) | 16 |
| **Case-level ASR (max-over-repeats)** | **45.7%** |
| Trial-level ASR (44 / 154 trials) | 28.6% |

**By attack group (case-level):**

| Group | Cases | Vulnerable | ASR |
|---|---:|---:|---:|
| Hijacking attacks | 15 | 12 | **80.0%** |
| Jailbreak | 20 | 4 | 20.0% |

**By family (case-level), most to least effective:**

| Family | Cases | Vulnerable | ASR |
|---|---:|---:|---:|
| Anti-DAN | 1 | 1 | 100% |
| DAN Jailbreak | 1 | 1 | 100% |
| DUDE | 1 | 1 | 100% |
| Long Prompt | 5 | 5 | 100% |
| Hate Speech | 5 | 4 | 80% |
| Violence Speech | 5 | 3 | 60% |
| Developer Mode | 2 | 1 | 50% |
| DAN | 13 | 0 | 0% |
| Image Markdown | 1 | 0 | 0% |
| STAN | 1 | 0 | 0% |

![ASR by attack family](asr_by_family.png)

**Trial outcome mix:** 44 injected · 56 off-target · 54 refused. Latency: median 6.4 s, max 112 s.

**Takeaway.** The model resisted every classic named **DAN** jailbreak (0% across 13 cases) — the payloads it has most likely been aligned against — but was far more exposed to **hijacking-style** injections (80% at the group level), with long-prompt and content-hijacking families landing most reliably. In other words, resistance to well-known jailbreak brands did **not** translate into resistance to injection generally.

### Smoke run

A 3-case functional check (DAN family, single repeat, direct vector) to verify the pipeline end to end — **not** a statistically meaningful benchmark.

| Metric | Value |
|---|---:|
| Cases | 3 |
| Injected | 0 |
| Outcome mix | 2 refused · 1 off-target |

All three cases were handled safely, confirming the harness runs and scores correctly before committing to the larger baseline.

> Figures above are derived directly from `Baseline_results.csv` and `Smoke_results.csv`. Re-running with a system profile (`--system-profile`) and both vectors (`--vector both`) would measure a specific deployment rather than the base model, and is the recommended next step.

---

## License

<!-- TODO: add your license (e.g. MIT, Apache-2.0) and a LICENSE file -->

## Author

[github.com/moustafa991982](https://github.com/moustafa991982)
