# Dev log

Key decisions and anything that went wrong, newest first.

## 2026-09-28: Kaggle quick checks: stand-in/smoke path collision (fixed)

Quick check 1 (tiny stand-ins on the T4) passed for both model families: data, processors, chat
templates, batched generation, scoring, and smoke-result writing all work on the GPU. The random
weights give 0 accuracy and chance-level AP, as expected.

Quick check 2 (SmolVLM2-500M, 32 real questions) stopped before answering anything. Stand-in and
`--limit` smoke runs shared `baseline/smoke/<model>/` paths, so the real-model run found the
stand-in's 16 answers in "its" predictions file. The header check refused to mix them, which is
exactly its job: nothing was scored. Fix: separate run kinds, `baseline/stand-in/` vs
`baseline/smoke/`, with a regression test that runs both back to back. The stale file on Kaggle
has to be removed once by hand (it's smoke data, nothing of value).

Harmless warnings seen on Kaggle and why: transformers 5.17 logs "Kwargs passed to
processor.__call__" from inside apply_chat_template even for a bare call (a test now pins that
the assistant turn is still opened); SmolVLM's config has an out-of-vocabulary pad_token_id
(generation gets the tokenizer's pad id explicitly); LFM uses the slower reference causal-conv1d
path because the CUDA kernel isn't installed. LFM's calibration batch will show whether that
matters for its GPU-hour estimate.

## 2026-09-28: Predictions out of git; on-device latency is prefill-bound

**Decision: prediction files never enter git.** Per-question predictions (raw and extracted
answers, P(unanswerable)) can repeat private details from the photos, e.g. text on mail or
medicine the model read aloud. The repo will become public and git history is permanent, so
"keep them in a private repo" is not enough. Now:
- They're stored in a **private Hugging Face dataset repo** (`cjitendr/lumen-predictions`,
  `predictions.repo` in pins). Each result in git records a pointer: path, Hub commit revision,
  sha256, line count. `lumen.hub.download` fetches at that revision and checks the sha256.
- Before any GPU time, the runner checks the repo exists, is **private**, and the token can write.
  An upload failure exits 5 with the error recorded; the rerun re-scores and re-uploads without
  re-answering. Smoke and failed runs are never uploaded (local disk only).
- Locally, predictions live outside `results/` and the Kaggle download folder, and are gitignored.
- `scripts/check_no_predictions.py` (stdlib only) blocks them as a pre-commit hook (`.githooks/`)
  and in CI, which scans tracked files **and the full history** (checkout uses `fetch-depth: 0`).
  It matches by name and by content (the predictions header), so renaming doesn't slip through.
  Verified: no predictions file was ever committed; history is clean.
- Needs from the user: create the private dataset repo; give the HF_TOKEN secret write access to
  it.

**On-device latency is prefill-bound (from the MLX smoke check).** SmolVLM2-500M spends ≈ 1.8–2.2 s
of a ≈ 2.3–2.5 s answer prefilling ~925 tokens, nearly all of them image tokens (the processor
tiles each photo into up to 16 × 512² tiles plus a global view, 64 tokens each), and only
~0.05–0.1 s decoding a short answer. Quantization mostly shrinks memory and speeds decoding; it
won't remove the prefill cost. **Stage 4 therefore measures image-resolution settings (image
budget / tile count) as a second axis next to quantization**, reporting accuracy and AP against
TTFT and memory for each combination. Lower resolution may hurt exactly the questions blind users
ask most (reading text), so the per-answer-type breakdown matters, and the setting that ships is
chosen without looking at val.

## 2026-09-28: Build-ahead during the Kaggle data run (Stage 2 code, MLX check)

**Mac shutdown.** The Mac overheated and shut down mid-session while this work was in progress.
Nothing had been committed, and the scratchpad was wiped. The only work that survived was the new
dependency groups in `pyproject.toml` / `uv.lock`, plus SmolVLM2-500M's weights in the HF cache
(downloaded at 02:09, just before the shutdown). Everything below was redone, and I committed
before the only heavier step (the MLX run). The MLX run itself was one warm-up plus 3 images,
and pmset recorded no thermal or performance warnings before or after.

**Notebook token leak (fixed).** The old clone cell put the token in git's URL and used
`check=True`: on a failed clone, `CalledProcessError`'s message prints the whole command, token
included. Sending git's output to DEVNULL didn't help, because the leak is in Python's
exception, not git's output. The new cell passes the token to git only as an HTTP auth header
through `GIT_CONFIG_*` environment variables for that one process, clears `GIT_TRACE*`, and
redacts git's error text. `tests/test_notebook_clone.py` runs the real cell against a stub git
that fails and echoes everything. Checked by mutation: the old cell fails the test with "token
leaked in exception message".

**wrapt warning.** Kaggle's image has a `sitecustomize.py` that imports `wrapt`, which our uv
venv lacks. setup.sh now builds the venv on a uv-managed Python (no sitecustomize) and drops
PYTHONPATH entries that contain one. **Not yet verified on Kaggle:** setup.sh prints which
sitecustomize the venv resolves, so the next session confirms it.

**GPU stack pins.** torch 2.14.0 from the **cu126** index (not cu130): CUDA 12.6 supports the
T4's sm_75 and runs on older drivers, avoiding the driver mismatch Strata hit with a CUDA 13
build. transformers 5.17.0, torchvision 0.29.0 (the fast image processors need it), num2words
(the SmolVLM processor needs it), accelerate 1.15.0. On the Mac, the PyPI CPU/MPS wheels (used
only for tests).

**Answerability score for zero-shot models.** P(the answer starts with "unanswerable"), summed
over four spellings (lower/capitalized × with/without a leading space; which one a model uses
depends on its chat template). It's exact, not an approximation. One prefill over the prompt and
image builds the KV cache; each spelling's remaining tokens are scored from a copy of the cache,
so the image is encoded once for scoring and once more inside `generate`. A test checks the
cached score against a full uncached forward per spelling.

**Answer extraction** (`lumen.zeroshot.extract`, versioned) keeps the first line, strips a leading
"Answer:" label and quotes, and maps whole-answer abstentions ("I can't tell", "cannot be
determined") to "unanswerable". Otherwise a model that abstains in its own words would be scored
wrong on exactly the questions it handled well. "unknown" is deliberately *not* mapped, since it
can be a real answer. The raw text is always stored next to the extracted answer.

**Tiny stand-ins instead of Hub "tiny-random" checkpoints.** The Hub's tiny SmolVLM and LFM2-VL
test checkpoints are internally inconsistent (their processor configs don't match their model
shapes; both crash in the vision projector). `tiny_stand_in` loads the pinned model's real config,
processor, and chat template (a few MB, no weights) and builds a ~5M-parameter random model with
the same structure. Tests on it prove left-padded batching doesn't change answers, the cached
scoring is exact, and the whole baseline stage runs end to end, before any GPU time is spent.

**Image tokens differ a lot between models** (from the pinned processors): SmolVLM2-256M/500M
resize to a 2048-px longest edge and tile into up to 17 tiles of 512² at 64 tokens each (~1.1k
tokens). LFM2.5-VL-450M uses up to 10 tiles plus a thumbnail at ≤256 tokens each (~2.9k).
SmolVLM2-2.2B uses 17 tiles of 384² at 81 tokens (~1.4k) with a 400M vision encoder. These go
into the GPU-hour priors.

**Stage 2 GPU-hour priors** (`run_stages.py baseline --dry-run`): dev, both prompts ≈ 5.2 h; val,
dev-chosen prompt ≈ 4.9 h; total ≈ 10.2 h, of which SmolVLM2-2.2B is 5.4 h. Conservative: model
load time is counted per run even though a model loads once per process. Calibration replaces
these with measured rates.

**MLX on the Mac works** (`scripts/mlx_smoke.py`, `results/mac/mlx_smoke_smolvlm2-500m.json`).
SmolVLM2-500M at its pinned revision through mlx-vlm 0.7.3 / MLX 0.32.2, weights as published
(float32, 507M parameters), M2 8 GB on battery:
- load 29 s (weights already cached); 1.89 GiB active after load; MLX peak 3.37 GiB.
- ~925 prompt tokens per question (image tiles + text). Warm: prefill ~430–505 tok/s, so time to
  first token ≈ 1.8–2.2 s; decode 63–100 tok/s; ≈ 2.3–2.5 s end to end for a short answer.
  The cold first call took 5.5 s (Metal kernel compilation).
- Implication for Stage 4: prefill of the image tokens dominates latency, not decoding. 4-bit
  weights will cut memory ~4× but may help prefill less than decode. Reducing image tiles is
  the bigger latency lever, and a trade-off to measure, not assume.
- 3 questions is a smoke check, not a benchmark: the accuracy there means nothing, and no
  answer text was stored (VizWiz photos can contain private details).

## 2026-09-28: Stage 1 data run on Kaggle

The `env`, `data`, and `mac_sample` stages ran on Kaggle (CPU session, Intel Xeon 2.2 GHz,
31 GB RAM, Python 3.12.13) at commit 48788b5. `data` ran twice by accident; both runs produced
identical hashes, counts, and splits, a useful check that the pipeline is deterministic. Both
result files are committed (`results/data/vizwiz_data.json`, `.1.json`).

**Hashes pinned.** The sha256 of Annotations.zip, train.zip, and val.zip are now in
`configs/pins.yaml` and enforced; `test_training_and_val_data_hashes_are_pinned` guards them.

**Counts.**

| Split | Questions | Answerable | Unanswerable | yes/no | number | other |
|---|---|---|---|---|---|---|
| train | 20,523 | 73.0% | 5,532 (27.0%) | 957 | 301 | 13,733 |
| val | 4,319 | 67.9% | 1,385 (32.1%) | 195 | 48 | 2,691 |

**Val has proportionally more unanswerable questions than train** (32.1% vs 27.0%, about 5
points, or 19% more relative). Consequences:
- Our dev split is carved from train, so it has train's rate (27%), not val's. A threshold or
  prompt tuned on dev meets a higher unanswerable base rate on val. Average precision depends on
  the base rate too (a random ranker scores AP ≈ the positive rate), so dev and val AP are not
  directly comparable; we compare models on the same split, never dev numbers against val numbers.
- A model that over-predicts "unanswerable" gains on val relative to dev. Val results are
  reported per answer type as well as overall, so a shift like this is visible.
- We do not reweight or resample dev to match val: that would use val's label distribution to
  make a choice, which our rules forbid.

**Splits.** train_fit 18,471 / dev 2,052 / Mac sample 300 (68.0% answerable, the same as val by
construction; 96 unanswerable, 187 other, 14 yes/no, 3 number). The ID lists are committed under
`configs/splits/` and are canonical from now on. The Mac sample archive (142 MB, sha256
`415d27e5…`, recorded in `results/mac_sample/`) is extracted to `data/mac_sample/` on the Mac and
kept out of git.

**Metric quirk 1 is small on VizWiz.** 812 of 205,230 train ground-truth answers (0.40%) and 206
of 43,190 val answers (0.48%) change under normalization, so they can never be matched exactly.
It's small, and it's the same for every model we compare.

**Unannotated images in the zips.** train.zip has 23,954 images for 20,523 questions; val.zip
has 7,750 for 4,319. That's 3,431 extra in each: the same count, which looked suspicious. Listing
the zips' central directories (names only, no image data) shows every file is a normally named
`VizWiz_{split}_*.jpg` in the right folder, with no macOS junk or files from other splits. The
image zips are shared with VizWiz's other tasks (e.g. captioning), which cover more photos than
VQA annotates. Harmless: Lumen only reads annotated images, and the data stage checks that every
annotated image is present.

## 2026-09-28: Stage 1

**Model choice.** The SmolVLM family's newest release is still SmolVLM2 (Feb 2025; nothing newer
under HuggingFaceTB on the Hub as of today). Fine-tune target: SmolVLM2-500M-Video-Instruct.
Zero-shot baselines: SmolVLM2-256M, SmolVLM2-2.2B, and LiquidAI LFM2.5-VL-450M (Apr 2026, the
newest strong model at this size; its lfm1.0 license restricts commercial use, so it is a
baseline only). All pinned by commit SHA.

**Dataset source.** The Hub mirror `lmms-lab/VizWiz-VQA` has only val and test, and no
`answerable` field, so Lumen uses the official vizwiz.org files. Sizes and ETags were recorded
from the host. The sha256s are trust-on-first-use from the first Kaggle `data` run, then pinned.

**Test answers are public.** VizWiz released test answers (`VQA_test.json`, 2026-04-28). Nothing
external stops a peek, so Lumen seals test itself: `lumen.eval.test_seal` allows exactly one
final evaluation (opt-in flag, clean tree, committed record, refuses reruns even after a crash).
The `data` stage doesn't download test at all.

**Official metric quirks, kept deliberately.** `src/lumen/eval/vizwiz_metric.py` ports the
official `vqaEval.py` (from the VizWiz API.zip, sha256 pinned) line for line, including two
behaviors that look like bugs. We keep them because the point of the official metric is
comparability with published VizWiz numbers. "Fixing" them would silently change every score.

1. *Answer-only normalization.* The prediction is normalized (punctuation, number words →
   digits, articles dropped, contractions), but the ten human answers are compared raw. A
   prediction of "two" becomes "2" and scores 0 against ten ground truths of "two". The `data`
   stage counts how many ground-truth answers are not in normalized form
   (`gt_answers_not_in_normalized_form`), so the size of the effect on VizWiz is measured, not
   guessed.
2. *Period-regex typo.* `periodStrip = "(?!<=\d)(\.)(?!\d)"` was meant to be the lookbehind
   `(?<!\d)`. As written, `(?!<=\d)` is a lookahead that is always true at a period, so only the
   "not followed by a digit" condition applies: "3." → "3", "3.5" kept. Same pattern copied.

Both are pinned by tests against golden outputs produced by running the official file
(`tests/fixtures/make_official_golden.py`).

**Images read from zips.** Extracting train.zip (11.3 GB) needs ~23 GB with the zip, over
Kaggle's ~20 GB persistent disk. `ZipImages` reads members directly and is fork-safe for
DataLoader workers.

**GPU quota.** Kaggle's API doesn't expose remaining weekly GPU quota, so GPU stages take
`--gpu-hours-left` from the user and refuse to start if estimate + 1 h reserve exceeds it.

**Splits.** Dev = 10% of train, stratified on answerable; Mac sample = 300 val questions
stratified on answerable × answer type. Selection ranks by sha256(seed:question_id), so it
doesn't depend on input order or library RNG versions. The committed ID lists are canonical.
