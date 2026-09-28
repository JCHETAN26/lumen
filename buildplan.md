# Lumen: Build Plan

**Lumen** is an on-device visual assistant for blind and low-vision users. The lean version is a
LoRA fine-tune of SmolVLM2-500M on VizWiz-VQA that answers questions about photos, says when a
photo can't answer the question, and runs 4-bit on a MacBook Air M2 (8 GB) with MLX. The
accessible app comes after.

**What it shows:** careful evaluation (the official metric, sealed test split, baselines,
variance), parameter-efficient fine-tuning under a tight GPU budget, selective answering
(answerability), and on-device deployment with measured accuracy and latency cost.

---

## Ground rules

- Every result comes from a script, records commit + hardware + versions, and has a baseline.
- No tuning on val or test. Dev (from train) makes every choice. Test is sealed: one final run.
- Datasets and models are pinned (sha256, commit SHAs) in `configs/pins.yaml`.
- Downloads, baselines, and training run on Kaggle. The Mac runs code, tests, and Stage 4 only.
- Every GPU stage estimates its GPU hours first and refuses to start past the weekly quota.
- Stop for review at the end of every stage.

## Hardware

| Machine | Specs | Role |
|---|---|---|
| MacBook Air | M2, 8 GB unified memory, fanless | Development; MLX quantization + on-device benchmarks (Stage 4) |
| Kaggle notebook | Tesla T4 16 GB (≈15 GB usable), fp16 only, ~30 GPU h/week quota, ~20 GB persistent disk | Downloads, baselines, fine-tuning, full evaluation |

## Models (pinned in `configs/pins.yaml`)

| Model | Role |
|---|---|
| SmolVLM2-500M-Video-Instruct | Fine-tuned and shipped |
| SmolVLM2-256M-Video-Instruct | Zero-shot baseline (smaller) |
| SmolVLM2-2.2B-Instruct | Zero-shot baseline (upper reference; also fits the Mac at 4-bit) |
| LFM2.5-VL-450M | Zero-shot baseline (newest non-SmolVLM model at this size; lfm1.0 license) |

## Data

| Split | Size | Use |
|---|---|---|
| train_fit | ~18.5k | Training |
| dev | ~2.05k (10% of train, stratified on answerable) | Every tuning choice |
| val | 4,319 | Reported results |
| test | 8,000 (answers public since 2026-04) | Sealed: one final evaluation |
| Mac sample | 300 from val (stratified on answerable × answer type) | On-device tests |

## GPU budget (priors; replaced by calibrated estimates when each stage runs)

| Stage | Work | Prior estimate |
|---|---|---|
| 1 | data, mac_sample | 0 GPU h (CPU sessions) |
| 2 | 4 models × (2 prompts on dev + 1 on val) | ~10.2 h (prior, `--dry-run`) |
| 3 | LoRA runs on 500M (≈18.5k examples × 2 epochs), 3 seeds for the final config, + ablation, + eval | ~15–20 h, spread over 1–2 weeks |
| Final | Test evaluation, one run | ~0.5 h |

---

## Stage 1: Repo, data pipeline, and evaluation code ✅

- [x] `CLAUDE.md`, `buildplan.md`, README (CI badge, VizWiz citation + CC BY 4.0, respectful-use
      note, metric quirks), `docs/devlog.md`.
- [x] uv project: core deps only (no torch); ruff; pytest; `uv.lock`.
- [x] GitHub Actions CI: ruff lint + format check + unit tests on every push.
- [x] `configs/pins.yaml`: model commit SHAs; dataset URLs, sizes, ETags; official eval-code sha256.
- [x] Official accuracy metric, ported line for line, tested against golden outputs of the official `vqaEval.py`.
- [x] Official answerability metrics (AP, F1 at 0.5) + dev-only threshold selection.
- [x] Strict annotation loader; deterministic stratified dev split + Mac sample.
- [x] Zip-backed image reader (no extraction; fits Kaggle's disk).
- [x] Pinned downloader: resume, size check, sha256 (trust on first use → committed).
- [x] Result writer with commit + hardware + versions; never overwrites.
- [x] GPU-quota guard (`lumen.budget`).
- [x] Test-split seal (`lumen.eval.test_seal`): opt-in flag, clean tree, one committed record, refuses reruns.
- [x] Live tests (Hugging Face, VizWiz host, Kaggle) that skip on usage caps.
- [x] Kaggle runner: token-safe clone, `setup.sh`, `env` / `data` / `mac_sample` stages, notebook.
- [x] Kaggle (CPU session): `env`, `data`, `mac_sample` ran at 48788b5 (2026-09-28).
- [x] Observed sha256s pinned in `configs/pins.yaml`; split ID lists in `configs/splits/`; results in `results/`.

**Done when:** CI is green; the data stage has run on Kaggle; hashes and splits are committed;
the data summary (counts, answerable rate, unmatchable-ground-truth count) has been reviewed.

## Stage 2: Zero-shot baseline on Kaggle (code done and tested; GPU runs not started)

Built ahead and tested end to end (fake backend in CI; tiny stand-ins with the real processors
locally): `lumen.zeroshot` + `run_stages.py baseline`. Prior GPU hours: dev ≈ 5.2, val ≈ 4.9,
total ≈ 10.2 (`--dry-run`). MLX on the Mac is confirmed early (`results/mac/mlx_smoke_*`).


- Add the GPU dependency group (torch, transformers, accelerate), pinned and verified on the T4.
- One inference harness for all four models: pinned revision, greedy decoding, max answer
  length, batching, fp16. Chat template per model family.
- Prompts: (a) plain "answer briefly", (b) + "If the image cannot answer the question, reply
  'unanswerable'." Prompt choice is made on **dev**, then reported on **val**.
- Answerability score for zero-shot models: P(first answer token(s) = "unanswerable"), so every
  model gets an average precision, not just a hard label.
- `baseline` stage behind the GPU guard: prior estimate → calibration on 64 dev questions →
  re-check → run. Resumable (per-question predictions appended to JSONL).
- Results: val accuracy (overall + per answer type), AP, F1, 95% bootstrap CIs, GPU hours
  estimated vs. used. Prediction files go to the private Hub dataset (never git); results record
  their revision + sha256. Guarded by `scripts/check_no_predictions.py` (pre-commit + CI).

**Done when:** four models × chosen prompt on val, with CIs and hardware recorded.

## Stage 3: LoRA fine-tuning and answerability detection

- LoRA (PEFT) on SmolVLM2-500M's language model, vision encoder frozen at first (unfreezing
  the connector is an ablation). Targets and rank chosen on dev.
- Training target: the most common human answer (the VizWiz convention; "unanswerable" for
  unanswerable questions), so the model learns to abstain in VizWiz's own vocabulary.
- **fp16 on the T4:** dynamic loss scaling (`torch.amp.GradScaler`); a NaN/Inf guard that skips
  a bad step, stops cleanly after N consecutive non-finite losses, keeps the last good
  checkpoint, and writes the stop reason into the result. Gradient checkpointing to fit memory.
- Answerability: same score as Stage 2 (P("unanswerable")); threshold for F1 chosen on dev.
- Ablation: with vs. without unanswerable examples in training (does abstention training cost
  accuracy on answerable questions?).
- Final config trained with 3 seeds; report mean ± std, plus CIs on val.
- Adapter pushed to a private Hub repo (`HF_TOKEN` secret scoped to that repo); commit SHA pinned
  in `configs/pins.yaml` `adapters:`.

**Done when:** fine-tuned vs. zero-shot 500M (and the other baselines) on val, with seeds and CIs;
adapter pinned.

## Stage 4: 4-bit quantization and on-device benchmarks (Mac)

- Add the MLX group (mlx, mlx-vlm). Download the pinned base model + adapter; merge; convert to
  MLX at fp16, 8-bit, and 4-bit (group size 64).
- On the 300-question Mac sample (checksummed archive from Kaggle): accuracy, AP, and agreement
  with the Kaggle fp16 predictions on the same IDs. The Kaggle numbers on these IDs are the
  baseline that isolates quantization's cost.
- Latency: time to first token, tokens/s, end-to-end per question; peak memory. Warm-up, then
  repeated runs; median and p90. Power source and throttle state recorded (fanless machine).
- **Image-resolution settings as well as quantization.** The early MLX check showed latency is
  dominated by prefilling ~925 image+text tokens (TTFT ≈ 1.8–2.2 s of ≈ 2.4 s), not by decoding.
  So compare the processor's image budget (max longest edge / number of 512² tiles, e.g. 2048 →
  1536 → 1024 → 512 px) against bits (fp16 / 8 / 4), as a grid: accuracy and AP on the Mac sample
  vs. TTFT and memory. Any resolution setting that ships is chosen on dev-derived data only
  (train-split images on Kaggle), never on val.
- Final test evaluation (sealed, one run) of the shipped configuration, run on Kaggle.

**Done when:** accuracy and latency tables for fp16 / 8-bit / 4-bit on the Mac, the final test
record committed.

## Stage 5: The accessible app

- React + TypeScript front end, talking to a local MLX inference server on the Mac.
- Screen-reader-first: capture or pick a photo, ask by voice or text, and get the answer spoken
  and as text. "I can't tell from this photo" with a short reason (blur, framing, lighting)
  when unanswerable.
- WCAG 2.2 AA, fully keyboard-operable, tested with VoiceOver; automated axe checks in CI.
- Scoped in detail at the start of the stage.
