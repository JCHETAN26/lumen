# Running Lumen on Kaggle

All dataset downloads, baselines, and training run in a **Kaggle notebook**, never on the Mac. The
Mac (M2 Air, 8 GB) gets only result files and a small pinned validation sample. This directory
is the runner:

| File | What it is |
|------|------------|
| `setup.sh` | Installs uv and the exact environment from `uv.lock`. Idempotent. |
| `run_stages.py` | Runs each stage, records machine and versions in every result, never overwrites results. |
| `notebook.ipynb` | Ready-made notebook with the cells below. |

**Stages**

| Stage | Build-plan stage | Accelerator | What |
|-------|------------------|-------------|------|
| `env` | 1 | any | Save the machine, GPU, and library versions |
| `data` | 1 | **None** | Download + verify pinned VizWiz files (~15 GB), validate annotations against the zips, build dev split + Mac sample IDs |
| `mac_sample` | 1 | **None** | Package the 300 pinned val images + annotations for the Mac |
| `baseline` | 2 | GPU T4 | Zero-shot baselines: 4 models × 2 prompts on dev, then each model's dev-chosen prompt on val |
| `baseline --long-edge N` | 3 prep | GPU T4 | Image resolution probe: SmolVLM2-500M on `dev_select` at 1,536 / 1,024 / 512 px |
| `resolution_choice` | 3 prep | **None** | Pick the resolution from the probe (rule fixed in advance) |
| `train_speed` | 3 prep | GPU T4 | 200 LoRA steps on `train_fit`: s/step, memory, tokens/s, projected h/epoch |
| `train`, `eval` | 3 | GPU T4 | LoRA fine-tuning and evaluation (added in Stage 3) |

## GPU quota guard

Kaggle's GPU quota is weekly and its API does not report what's left. Every GPU stage therefore
takes `--gpu-hours-left H` (read it from kaggle.com → Settings), estimates its own GPU hours
before loading anything, re-estimates from a short timed calibration batch, and **refuses to start**
if the estimate plus a 1-hour reserve exceeds `H` (`lumen.budget`). The estimate and the hours
actually used are both recorded in the stage's result. CPU stages warn if a GPU is attached.

## The test split is sealed

VizWiz's test answers are public, so Lumen seals them itself (`lumen.eval.test_seal`). Test
answers load only inside the single final evaluation, which needs `--final-test-evaluation` and a
clean commit, and writes `results/test/FINAL_TEST_RUN.json`. Once that file exists, test
evaluation refuses to run again, even after a crash. The `data` stage does not download test
data at all.

## 1. One-time setup

1. A Kaggle account with **phone verification** (needed for GPU and internet).
2. A **GitHub fine-grained token** with *Contents: read* on `JCHETAN26/lumen` only.
3. *(Stage 2)* Private Hugging Face repos `cjitendr/lumen-predictions` (dataset) and
   `cjitendr/lumen-adapter` (model, for Stage 3), and a **fine-grained HF token** with write
   access to those two only (nothing account-wide). `run_stages.py storage_check` verifies both. Prediction files are stored there,
   never in git; the runner refuses to start if the repo is missing, public, or not writable.

## 2. Create the notebook

1. kaggle.com → **Create → New Notebook**, then **File → Import Notebook** → `kaggle/notebook.ipynb`.
2. **Settings:**
   - **Accelerator: None** for `data` and `mac_sample`; **GPU T4** from Stage 2 on (the runner uses one GPU).
   - **Internet: On.**
   - **Persistence: Variables and Files**, so the 15 GB of zips and partial downloads survive a
     restart (downloads resume from where they stopped).
3. **Add-ons → Secrets:** add `GITHUB_TOKEN` and (from Stage 2) `HF_TOKEN`, and attach them.

## 3. Cells

**Clone or update:** the first code cell of `notebook.ipynb`. That cell is the only copy, so
the code that runs is the code under test. The token:

- is read from the `GITHUB_TOKEN` secret inside a helper and passed to git only as an HTTP
  `Authorization` header, via `GIT_CONFIG_*` environment variables set for that one git process;
- never appears in a command line or URL, so a failed clone can't leak it through an exception
  message (the old cell could: `check=True` puts the full command, token included, into the
  `CalledProcessError` traceback);
- is redacted from git's error text before it's shown, and `GIT_TRACE*` variables are cleared;
- is never written to disk: the stored remote is the plain URL.

`tests/test_notebook_clone.py` runs the real cell against a stub git that fails, then succeeds,
and checks that the token appears nowhere: not in output, exceptions, tracebacks, argv, or files.

If the cell fails, the error shows git's (redacted) message; check the token's scope and expiry.

**Setup, then stages:**

```python
%cd /kaggle/working/lumen
!bash kaggle/setup.sh
!source /kaggle/working/lumen-env.sh && uv run python kaggle/run_stages.py env
!source /kaggle/working/lumen-env.sh && uv run python kaggle/run_stages.py data
!source /kaggle/working/lumen-env.sh && uv run python kaggle/run_stages.py mac_sample
```

## Stage 2: zero-shot baselines

```bash
B="uv run python kaggle/run_stages.py baseline"
$B --model smolvlm2-500m --split dev_select --prompt both --dry-run --gpu-hours-left H  # plan
$B --model smolvlm2-500m --split dev_select --prompt both --gpu-hours-left H   # prompt choice
$B --model smolvlm2-500m --split val --prompt dev-selected --gpu-hours-left H  # val
uv run python kaggle/run_stages.py baseline_summary
```

- **Prompt choice on `dev_select`**, a fixed stratified 512-question subset of dev (approved
  2026-09-30; built deterministically on Kaggle from the committed dev IDs and pinned seed,
  written to the download folder's `splits/`, then committed and enforced like the other splits).
- **Prefetching on** (2 workers, the default): measured 0.666 s/question vs 0.798 off.
  `--prefetch-workers 0` turns it off; `auto` re-measures both on the first batches.
- **Prior GPU hours** (`lumen.zeroshot.estimates`, 1.3x safety; SmolVLM2-500M measured, the
  others scaled x1.48 until their own calibration batches measure them):

  | Model | s/question | prompt choice (512 x 2) | val (4,319) | total |
  |---|---|---|---|---|
  | SmolVLM2-500M | 0.67 (measured) | 0.36 | 1.15 | 1.51 |
  | LFM2.5-VL-450M | 0.67 | 0.36 | 1.15 | 1.51 |
  | SmolVLM2-256M | 0.56 | 0.29 | 0.96 | 1.25 |
  | SmolVLM2-2.2B (next week) | 2.29 | 1.06 | 3.79 | 4.85 |

  Each run re-estimates from a 32-question calibration batch and stops cleanly if the measured rate
  no longer fits; answered questions are kept and a rerun resumes.
- **Val uses the dev-chosen prompt only** (`--prompt dev-selected`: best `dev_select` accuracy,
  ties by AP). Explicit prompts on val are refused.
- **fp16 on the T4:** non-finite logits stop the run and write a `failed_non_finite` result; nothing
  is scored silently.
- **Outputs:** `baseline/<model>/<split>-<prompt>.json`: aggregate metrics with bootstrap CIs,
  GPU hours estimated vs. used, full run metadata, and a **pointer** to the predictions (Hub repo,
  path, commit revision, sha256). Commit these to `results/baseline/`.
- **Predictions never enter git.** Per-question outputs (raw and extracted answers, P(unanswerable))
  can repeat private details from users' photos, this repo will be public, and git history is
  permanent. Completed runs upload `.predictions.jsonl` to the private Hub dataset; smoke and failed
  runs keep them on Kaggle's disk only. `lumen.hub.download(pointer)` fetches one at its pinned
  revision and verifies its sha256. If an upload fails, the stage exits 5 and a rerun re-uploads
  without re-answering anything.

## Stage 3 prep: image resolution

```bash
B="uv run python kaggle/run_stages.py baseline --model smolvlm2-500m"
$B --split dev --prompt abstain --gpu-hours-left H                     # 2,048 px (default)
for le in 1536 1024 512; do
  $B --split dev --prompt abstain --long-edge $le --gpu-hours-left H
done
uv run python kaggle/run_stages.py resolution_choice      # CPU; commit its result
$B --split val --prompt dev-selected --long-edge chosen --gpu-hours-left H
```

- **Why:** image tokens are most of SmolVLM2's cost. The processor rounds the long edge up to a
  multiple of its 512 px tile and splits it into tiles plus a global view, 64 tokens each; a 3:4
  portrait photo gets 13 / 10 / 5 / 1 tiles at 2,048 (default) / 1,536 / 1,024 / 512 px, i.e. 832
  of ~875 prompt tokens at the default. 768 px gives the same tiles as 1,024, so it isn't offered.
  Smaller photos are upscaled, so the tile count depends on the aspect ratio, not the photo's size.
- **On all 2,052 dev questions**, not dev_select: the paired CI needs the sample size. Prior cost
  0.60 + 0.49 + 0.30 + 0.28 = 1.68 GPU h. The 2,048 px arm is a full-dev run at the default
  resolution (the existing 2,048 px results cover dev_select only).
- **Rule, fixed 2026-10-03 before any probe result** (`lumen.zeroshot.resolution`): the best arm
  has the highest dev accuracy; the chosen arm is the smallest long edge whose 95% paired bootstrap
  CI on (accuracy − best accuracy) has a lower bound ≥ −2.0 points. The margin is a product
  decision: 1,024 px uses ~2.6x fewer image tokens, cutting training cost and on-device latency.
  AP is reported with its own paired CI, not decisive. `resolution_choice` re-scores every arm from
  its predictions on the same questions, and also reports each arm's difference from 2,048 px.
- **Matched baseline:** `--long-edge chosen` reads the committed `resolution_choice` result; the
  fine-tune is compared against SmolVLM2-500M zero-shot on val at that resolution. If 2,048 px wins,
  the existing val run already is that baseline and the command runs nothing.
- **Files:** a capped run is `<split>-<prompt>-le<N>.json` with its own predictions file, whose
  header records the cap; default-resolution runs keep their names and headers. Prompt choice only
  reads default-resolution runs.

## Stage 3 prep: training speed test

```bash
uv run python kaggle/run_stages.py train_speed --long-edge chosen --dry-run
uv run python kaggle/run_stages.py train_speed --long-edge chosen --gpu-hours-left H
```

- 200 optimizer steps of LoRA (rank 16 on the language model's attention + MLP projections; the
  vision encoder stays frozen) on `train_fit`, batch 4 x 2 accumulation, at the chosen resolution,
  with the prompt chosen on dev_select. Targets: the most common human answer; `unanswerable` for
  unanswerable questions (`lumen.train.targets`).
- fp16 on the T4: frozen base in fp16, LoRA weights in fp32 under autocast, dynamic loss scaling.
  A non-finite loss skips the step; 10 in a row stop the run (exit 4) with the last good
  checkpoint kept and the stop recorded. Calibrated after 10 steps; stops cleanly (exit 3) if the
  rest no longer fits the quota.
- Prior: ~5.4 s/step at 2,048 px, ~2.1 s at 1,024 (scaled from the measured inference rate), so
  0.37-0.60 GPU h for the test. The result records s/step (median, p90 after 5 warm-up steps),
  peak memory, tokens/s, and projected hours per epoch (2,309 steps for 18,471 examples).
- Checkpoints go to `/kaggle/working/lumen-checkpoints/` only; the adapter isn't published.

## 4. What to download, and where it goes

Only **`/kaggle/working/lumen-results/`** leaves Kaggle (Output tab). Commit its contents on the
Mac. Copy only files that don't exist yet; never overwrite a committed result.

| Download | Commit to |
|----------|-----------|
| `environment.json` | `results/kaggle/` |
| `vizwiz_data.json`, `observed_hashes.json` | `results/data/` |
| `splits/*.txt` | `configs/splits/` (these become the pinned splits) |
| `mac_sample/*.tar.gz` | **not committed**: extract to `data/mac_sample/` on the Mac (gitignored) |
| `mac_sample_val_*.json` | `results/mac_sample/` (records the archive's sha256) |
| `baseline/<model>/*.json` | `results/baseline/<model>/` (`baseline/smoke/` too, if you want the checks kept) |
| `stage3_prep/*.json` | `results/stage3_prep/` |

After the first `data` run, copy the sha256 values from `observed_hashes.json` into
`configs/pins.yaml`. From then on, any download that doesn't match stops the run.
