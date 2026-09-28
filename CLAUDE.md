# Lumen

An on-device visual assistant for blind and low-vision users. The lean version fine-tunes
**SmolVLM2-500M** with LoRA on **VizWiz-VQA**, teaches it to recognize unanswerable photos using
VizWiz's answerability labels, quantizes it to 4 bits, and runs it on a MacBook Air (M2, 8 GB)
with MLX. An accessible React/TypeScript app comes after the lean version.

This is a portfolio project. Correct evaluation, honest baselines, and reproducible results
matter more than feature count. The phase-by-phase plan is in `buildplan.md`; check it before
starting new work. Key decisions and anything that went wrong go in `docs/devlog.md`.

## Machines

| Machine | Role | Never |
|---|---|---|
| **MacBook Air M2, 8 GB** (fanless; throttles under sustained load) | Writing code, unit tests, Stage 4 MLX quantization and on-device benchmarks on the pinned Mac sample | Download datasets, train, or run baselines |
| **Kaggle notebook, Tesla T4 (15 GB, no bf16, no FlashAttention-2)** | All dataset downloads, zero-shot baselines, LoRA fine-tuning, full-split evaluation | Store secrets in the repo, print tokens |

- Code moves Mac → GitHub (private `JCHETAN26/lumen`) → Kaggle via `kaggle/` (see `kaggle/README.md`).
  The notebook clones with a GitHub token from Kaggle Secrets; the token is never printed and
  never written to disk.
- Only result files come back from Kaggle, plus the ~300-image pinned Mac sample. LoRA adapters
  go to a private Hugging Face Hub repo (token: Kaggle secret scoped to that repo) and are pinned
  by commit SHA in `configs/pins.yaml`.
- **Kaggle GPU quota:** every GPU stage estimates its GPU hours first (`lumen.budget`) and refuses
  to start if the estimate plus a 1 h reserve exceeds `--gpu-hours-left`. CPU-only stages
  (`data`, `mac_sample`) run with Accelerator: None.

## Pins: datasets and models

- `configs/pins.yaml` is the single source of truth: every model by **full commit SHA**, every
  dataset file by URL, size, and **sha256**. Code loads pins through `lumen.pins`. Never download
  an unpinned artifact or use a branch name (`main`) as a revision.
- Dataset: the official VizWiz files from vizwiz.org, not a Hub mirror (the Hub mirror lacks
  train and answerability labels). A null sha256 is trust-on-first-use: the first Kaggle `data`
  run records it, and then it is committed. After that, a mismatch stops the run.
- Split ID lists are committed under `configs/splits/`; later stages read those files.

## Splits: no tuning on test data

| Split | Source | Use |
|---|---|---|
| `train_fit` | official train minus dev | Training only |
| `dev` | 10% of official train, stratified on answerable | **All** choices: hyperparameters, prompts, checkpoint, unanswerable threshold |
| `val` | official val (4,319) | The reporting split. Never used to choose anything. |
| `test` | official test (8,000; answers public since 2026-04-28) | **Sealed.** One final evaluation, ever. |

- The test seal (`lumen.eval.test_seal`): test answers load only through a `FinalTestRun`, which
  requires `--final-test-evaluation` and a clean git tree, and writes
  `results/test/FINAL_TEST_RUN.json` (commit, dates, results). Once that record exists, test
  evaluation refuses to run, including after a crash. Commit the record immediately. Never
  delete or edit it to get another run.
- If a number on val looks worth tuning toward, the tuning happens on dev and val is re-reported.

## Evaluation: the official VizWiz metrics

- **Accuracy:** `lumen.eval.vizwiz_metric` is a line-for-line port of the official `vqaEval.py`
  (sha256 in pins): the mean over the 10 leave-one-annotator-out subsets of
  `min(1, matches / 3)`. It is tested against outputs of the official file
  (`tests/fixtures/official_vqaeval_golden.json`).
- **Answerability:** average precision with *unanswerable* as the positive class, plus the
  official F1 at 0.5 (`lumen.eval.answerability`). F1 at a dev-tuned threshold is labeled as such.
- **Two quirks of the official code are kept deliberately** so numbers are comparable with
  published VizWiz results. Don't "fix" them:
  1. Only the prediction is normalized; ground-truth answers are compared raw.
  2. The period regex `(?!<=\d)` is a typo for `(?<!\d)`, and we copy it as written.
- A missing prediction is an error, never a silent zero.

## Results

- Every result is produced by a script and written with `lumen.runmeta.write_result`, which
  records git commit and dirty flag, hardware (GPU name, driver, and memory on Kaggle; chip, RAM,
  macOS, power source, and throttle state on the Mac), Python and library versions, and the
  timestamp. It never overwrites an existing result. No hand-copied numbers.
- Every comparison has a baseline: zero-shot SmolVLM2-256M, SmolVLM2-2.2B, LFM2.5-VL-450M and
  SmolVLM2-500M itself for the fine-tune; fp16 on the same questions for quantization; the same
  question IDs on both sides of every comparison.
- Report variance: multiple seeds where training is involved, bootstrap confidence intervals on
  accuracy and average precision, and median and p90 over repeated runs for latency.
- Report weaknesses honestly, including where a baseline wins.

## Testing

- `uv run pytest -m "not live"`: unit tests, no network. This is what CI runs on every push.
- Live tests (`@pytest.mark.live`, `tests/live/`) call Hugging Face, Kaggle, the VizWiz host, and
  later Anthropic (only if an LLM judge is added). They **skip cleanly on usage caps and rate
  limits** (`lumen.live.usage_cap_reason` via `tests/conftest.py::skip_on_usage_cap`) and on
  missing credentials; any other error fails. Skip reasons print (`-rs`).
- New metric or data code gets unit tests with edge cases. Anything touching the metric keeps
  the official-golden tests passing.

## Code conventions

- Python 3.11+, uv (`uv.lock` committed), ruff (lint + format, line length 100), pytest, type hints.
- Dependency groups: core (metric and data, no torch, used by CI), GPU groups for Kaggle
  (Stage 2), and MLX for the Mac (Stage 4). Keep torch out of core.
- Fail loudly: strict validation of annotations, pins, and inputs, with errors that say what to do.
- fp16 training on the T4 uses dynamic loss scaling and a NaN/Inf guard that stops cleanly,
  keeps the last good checkpoint, and records the stop in the result (Stage 3).

## Data ethics

VizWiz photos were taken by blind people asking for help and can contain private information.
Never commit, display, or redistribute them, and never quote private details from them. Model
prediction files (`*.predictions.jsonl`) can repeat such details: they stay in this private repo
and are never quoted publicly; on-device smoke checks store scores, not answer text. See the
README's "Respectful use" section. The dataset is CC BY 4.0: cite Gurari et al., CVPR 2018.

## Secrets

- Never commit secrets. `.env` and `kaggle.json` are gitignored.
- Kaggle Secrets: `GITHUB_TOKEN` (Contents: read on this repo) and `HF_TOKEN` (write, scoped to
  the adapter repo).

## Workflow

- Work in the stages of `buildplan.md`, and stop for review at the end of each stage.
- Small, tested commits. Propose a plan before large changes; ask when a design choice is unclear.
- GitHub: `JCHETAN26/lumen` (private), default branch `main`.
