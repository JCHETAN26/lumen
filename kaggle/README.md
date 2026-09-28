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
| `baseline` | 2 | GPU T4 | Zero-shot baselines (added in Stage 2) |
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
3. *(Stage 3)* A **Hugging Face fine-grained token** with write access to the private adapter
   repo only (not account-wide), used to push LoRA adapters.

## 2. Create the notebook

1. kaggle.com → **Create → New Notebook**, then **File → Import Notebook** → `kaggle/notebook.ipynb`.
2. **Settings:**
   - **Accelerator: None** for `data` and `mac_sample`; **GPU T4** from Stage 2 on (the runner uses one GPU).
   - **Internet: On.**
   - **Persistence: Variables and Files**, so the 15 GB of zips and partial downloads survive a
     restart (downloads resume from where they stopped).
3. **Add-ons → Secrets:** add `GITHUB_TOKEN` (and, from Stage 3, `HF_TOKEN`) and attach them.

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

After the first `data` run, copy the sha256 values from `observed_hashes.json` into
`configs/pins.yaml`. From then on, any download that doesn't match stops the run.
