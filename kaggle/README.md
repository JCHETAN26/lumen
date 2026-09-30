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

After the first `data` run, copy the sha256 values from `observed_hashes.json` into
`configs/pins.yaml`. From then on, any download that doesn't match stops the run.
