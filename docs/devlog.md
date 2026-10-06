# Dev log

Key decisions and anything that went wrong, newest first.

## 2026-10-05: Resolution probe: the rule picked 512 px; overridden to 1,024 px (deviation)

**Probe** (SmolVLM2-500M zero-shot, abstain prompt, all 2,052 dev questions;
`results/stage3_prep/resolution_choice*.json` and `results/baseline/smolvlm2-500m/dev-abstain*.json`):

| Long edge | Accuracy [95% CI] | AP | Said unanswerable | s/question (T4) |
|---|---|---|---|---|
| 2,048 | 30.14 [28.34, 31.89] | 47.88 | 10.5% | 0.648 |
| 1,536 | 30.63 [28.86, 32.26] | 48.79 | 11.1% | 0.511 |
| 1,024 | 31.88 [30.09, 33.65] | 49.06 | 11.4% | 0.293 |
| 512 | 40.01 [38.00, 41.86] | 44.88 | 30.2% | 0.132 |

The pre-registered rule chose **512 px**: it is the best arm, and every other arm's paired CI vs it
is 6-12 points below (far outside the 2-point margin). The probe used 0.91 GPU h (prior 1.68).

**Why 512 won: abstention, not seeing.** On answerable questions only (n = 1,499; reported by
`resolution_choice` as `answerable_breakdown`):

| Long edge | Says "unanswerable" | Credit from "unanswerable" | Credit from real answers | Accuracy when it answers |
|---|---|---|---|---|
| 2,048 | 5.7% | 2.52 | 28.42 | 30.13 |
| 1,536 | 6.1% | 2.72 | 28.47 | 30.31 |
| 1,024 | 6.3% | 2.74 | 30.09 | 32.10 |
| 512 | 21.2% | 9.18 | 24.58 | 31.19 |

At 512 px (one 64-token tile) the model abstains three times as often, which earns credit on the
27% of dev that is unanswerable (accuracy there 57 vs 28-29) and partial credit on answerable
questions that some annotators marked unanswerable. Its credit from real answers is the lowest of
all arms, and AP is the lowest too (it abstains more, not more accurately). By answer type 512 is
also worst on number (37 vs 48 at 1,024) and yes/no (44 vs 50), best only on other (33.0 vs 31.4).

**Decision (2026-10-05, after seeing dev results): use 1,024 px.** On the questions the model
answers, 1,024 px is best, with 2.6x fewer image tokens than 2,048. Fine-tuning teaches abstention
directly, so 512's zero-shot edge isn't expected to survive training, while its lost image detail
would: a bad trade for users who need labels and small text read. This is a **deviation from the
pre-registered rule**, recorded in `configs/resolution_override.yaml`; `--long-edge chosen` honors it
and refuses if the rule's recorded choice changes. Only dev data was used; val and test were not
looked at. The rule's flaw, in hindsight: zero-shot accuracy rewards abstention, so a resolution
rule should have used credit from real answers (or AP) alongside accuracy. Reported as a weakness.

**Also:** the Kaggle machine restarted mid-session (working directory reset, GPU venv in /tmp
lost, files in /kaggle/working kept), so cells 1 -> 2 -> 3 must be rerun after any restart.

## 2026-10-05: SmolVLM2-500M result files lost with the Kaggle session; rebuilt from the Hub

**What went wrong.** The SmolVLM2-500M runs (prompt choice on dev_select, then val) completed on
Kaggle and uploaded their predictions to the private Hub dataset, but the result files and the
generated `dev_select.txt` were never downloaded. The session's disk was then wiped:
`/kaggle/working/lumen-results/`, the local predictions, and the clone's `results/` were all gone
(checked 2026-10-05), and the download on hand was an older Stage 1 one, identical to what was
already committed. The numbers in the planning notes (dev_select: abstain 28.79 vs plain 24.92; val
32.07 [30.7, 33.3], AP 52.67) came from notebook output, not from a committed result.

**Recovery, no GPU.** New CPU stage `rebuild_baseline` re-creates the result files from the Hub
predictions (details in kaggle/README.md). Results are deterministic given the predictions, so the
metrics are exactly what the original runs computed; what can't be recovered (GPU hours, timing,
batch settings) is recorded as lost, never estimated. If the Hub files turn out to be missing or
incomplete, the two runs are rerun (~1.5 GPU h).

**Also:** `resolution_choice` and the rebuild load annotations only (`_load_split(images=False)`),
so neither re-downloads or re-verifies the 11 GB train zip.

**Lesson:** download `lumen-results` at the end of every session, before anything else.

## 2026-10-03: Stage 3 training code and the speed test

**Targets: the most common human answer** (VizWiz convention; exact ties alphabetical), and the
literal `unanswerable` for unanswerable questions. A metric-aware alternative was drafted and
rejected before any training: choose whichever human answer the official metric scores highest.
Forms of one answer ("two" / "2") always tie under it, because the prediction is normalized, so it
can only differ from the majority by switching to a *different* minority answer (6x "the dog" vs 4x
"cat" picks "cat", since "the dog" is scored as "dog" against raw ground truths). That teaches
wrong content for metric credit. Instead, each run counts the targets the metric can't credit
(`zero_score_targets`) and tied majorities, so the quirk's effect on training is measured.

**Loss on the answer only.** Labels cover the answer tokens and the `<end_of_utterance>` that ends
the turn; prompt, image, and padding tokens are -100. The answer's start is located by the chat
template's generation marker in the token ids, and a test on the real processor checks it equals
the length of the prompt encoded alone, i.e. exactly where generation starts at inference.

**LoRA and fp16.** Rank 16, alpha 32, dropout 0.05 on the language model's q/k/v/o and MLP
projections (the regex is anchored on `text_model`: the vision encoder has q/k/v_proj too, and a
check refuses any trainable weight outside the language model). These are defaults for the speed
test; rank, targets, and learning rate are chosen on dev in Stage 3. The base stays fp16; LoRA
weights are fp32 because GradScaler can't unscale fp16 gradients. Gradient accumulation weights
each micro-batch's mean loss equally (not by token count). Checkpoints are written only after good
steps, atomically, with optimizer/scaler/scheduler/RNG state; the data order is a pure function of
(seed, epoch), and a test checks that a resumed run's weights equal an uninterrupted run's.

**Tested on CPU only so far.** The loop, NaN guard, resume, and the whole `train_speed` stage run
in tests with a tiny random-weight stand-in in fp32. The fp16 autocast + loss-scaling path cannot
run on CPU; the T4 speed test is its first real run, and its overflow and non-finite counts are in
the result. `peft==0.21.2` is added to the GPU group.

## 2026-10-03: Image resolution probe (Stage 3 prep)

**What the processor actually does** (SmolVLM2-500M's processor at the pinned revision, run on
CPU): the long edge is rounded *up* to a multiple of the 512 px tile, the image is split into tiles
plus one global view, 64 tokens each. A 3:4 portrait photo, the usual VizWiz shape, gets 13 / 10 /
5 / 1 tiles at a 2,048 (default) / 1,536 / 1,024 / 512 px cap, so at the default 832 of ~875 prompt
tokens are image tokens (not ~1,100, which assumed a square photo). Small photos are upscaled to the
same grid, so the tile count follows the aspect ratio, not the photo's size. A 768 px cap gives the
same tiles as 1,024, so the probe arms are 2,048 / 1,536 / 1,024 / 512.

**Rule, fixed 2026-10-03 before any probe result** (`lumen.zeroshot.resolution`): the *best* arm is
the one with the highest accuracy on **all 2,052 dev questions** (exact ties: fewer image tokens).
The *chosen* arm is the lowest-token resolution (smallest long edge) that is not meaningfully worse
than the best: the lower bound of the 95% paired bootstrap CI on its accuracy difference from the
best is at least **-2.0 points** (the non-inferiority margin). AP is reported with its own paired CI
but does not decide.

**Why a 2-point margin (product decision, made before any result).** 1,024 px uses about 2.6x fewer
image tokens than 2,048 px (320 vs 832 for a portrait photo), which cuts training cost and
on-device latency, both central to Lumen (an 8 GB fanless MacBook Air). Accepting up to 2 accuracy
points for that is a stated trade-off, not something read off the data.

**Why all of dev, not dev_select.** The CI is paired, so its width depends on how many questions
change score between two resolutions (d), roughly half-width ≈ 1.96 · 100·√d / n with 0/1 scores.
A first version (2026-10-03, never run) used the 512-question `dev_select` with a 1-point margin; a
cheaper arm could then pass only if it changed at most ~7 of 512 outcomes, so the rule would almost
always have kept the best arm. With n = 2,052 and a 2-point margin, an arm with the same point
accuracy passes if up to ~440 questions (~21%) change score, and one 1 point worse if up to ~110
(~5%) do. VizWiz's fractional scores loosen this a little. Earlier drafts, both before any probe
result: "highest accuracy wins" (2026-10-01), then the 1-point margin on dev_select.

**Cost.** Four full-dev runs (2,048 / 1,536 / 1,024 / 512 px): prior 0.60 + 0.49 + 0.30 + 0.28 =
1.68 GPU h (1.3x safety, model load per run), up from 0.51 h on dev_select. The 2,048 px arm is a
new full-dev run at the default resolution: the existing 2,048 px results cover only dev_select
(512) and the quick checks' smoke runs. If an earlier full-dev `dev-abstain` predictions file is on
Kaggle's disk with the same header, the run resumes from it.

Limitation: zero-shot accuracy is a proxy for how well the *fine-tuned* model uses a resolution.

**How:** `baseline --long-edge N` caps the processor's long edge (SmolVLM2-500M/256M only; LFM2.5
tiles differently and is refused). The cap goes into the predictions header only when set, so the
default-resolution files already on the Hub stay valid. The probe uses the prompt chosen on
dev_select (abstain); a full-dev run of one prompt doesn't change prompt choice, which needs both
prompts on a split. `resolution_choice` (CPU) re-scores each arm from its predictions on the same
2,052 questions; the val run at the chosen resolution reads the committed choice
(`--long-edge chosen`). Prior cost scales with portrait tiles, floored at 0.35x, until the
calibration batch measures it.

## 2026-09-30: Plan for 12 GPU h: prompt choice on dev_select; prefetch on

**Decisions (approved).** (1) Zero-shot prompts are chosen on `dev_select`, a fixed stratified
(answerable x answer type) 512-question subset of dev, instead of all 2,052 dev questions: still
dev-only, a paired comparison on the same questions, saving ~5 GPU h. Stratifying needs labels,
and annotations stay on Kaggle, so the runner builds the subset there deterministically (committed
dev IDs + pinned seed), writes it to the download folder, and requires any committed copy to match
exactly; it's committed after the first Kaggle run. (2) SmolVLM2-2.2B moves to next week.
(3) This week also builds the image-resolution probe and the Stage 3 training speed test;
resolution is chosen on dev data only, and SmolVLM2-500M zero-shot is re-run on val at the chosen
resolution so the fine-tune comparison is like for like.

**Prefetching measured on the T4:** 0.666 s/question with 2 workers vs 0.798 without (different
64-96-question samples, so approximate), about 17% faster. Default is now 2 workers; the
estimates use the prefetch-on rate (SmolVLM2-500M 0.67; others scaled x1.48: 256M 0.56,
LFM 0.67, 2.2B 2.29). `--prefetch-workers auto` exists (warm-up batch, then 4 batches on and 4
off, keep the faster) but isn't the default.

**This week (≈ 12 GPU h):** SmolVLM2-500M, LFM2.5-VL-450M, SmolVLM2-256M prompt choice + val
≈ 4.3 h; Stage 3 prep ≈ 1.5 h (rough); SmolVLM2-500M val at the chosen resolution ≈ 0.5-0.8 h.
That leaves ≈ 5 h of margin. The prefetch saving (≈ 0.7 h) goes to margin rather than moving
SmolVLM2-2.2B's val run (3.8 h) into this week, which would leave too little for estimate error
and the runner's 1 h reserve.

## 2026-09-30: Measured speed, updated GPU-hour estimates, warnings resolved

**Quick check 2 on Kaggle** (SmolVLM2-500M zero-shot, dev, abstain prompt, fp16): accuracy ~30%,
AP 44-52, said-unanswerable ~10% vs ~27% truly unanswerable, no NaN. Measured **0.67-0.80
s/question** across the prefetch on/off A/B, vs. the 0.45 prior. Which rate belonged to which
setting wasn't reported, so the estimates use the slower 0.80.

**Estimates updated** (`lumen.zeroshot.estimates`): SmolVLM2-500M 0.80 measured; the other three
scaled by the same factor (x1.78): 256M 0.68, LFM2.5-VL-450M 0.80, 2.2B 2.76 s/question. The
proportional scaling assumes their mix of CPU preprocessing and GPU work is similar to 500M's.
That's least true for 2.2B (more GPU-bound) and LFM (different tiling, reference conv kernels),
so their first calibration batches may move them either way. With 1.3x safety: dev (both prompts,
full dev) 8.0 h + val 8.4 h ≈ 16.4 h for all four models. That doesn't fit the 12 h left this week.

**pad_token_id warning: harmless.** The out-of-range 128002 is only in SmolVLM's *top-level*
config (a stale value). Everything that handles padding uses token id 2 (`<|im_end|>`, valid):
text_config.pad_token_id, the published generation config, the tokenizer, and our `generate`
call, which passes the tokenizer's pad id explicitly (since 299c367). Left padding + attention
masks make padding content irrelevant anyway; the batching-invariance test checks exactly that.

**processor_kwargs warning: gone.** Filtered since 9ab88f0; a real stand-in run through
HFBackend logs zero of them. Kaggle ran code at or after 5dbdfee (its S2.5 cell uses
`--prefetch-workers`, which only exists from 5dbdfee), so it had both fixes.

## 2026-09-28: Stage 2 quick checks passed; prefetching, and a correction

**Quick checks on the T4.** Storage check: both private HF repos are private and writable, which
also confirms the fine-grained token scope parsing works against a real token. Quick check 2
(SmolVLM2-500M, 32 dev questions, fp16): finite logits, accuracy 25.0 [10.6, 40.3], AP 66.0,
9.4% said unanswerable against ~27% unanswerable in dev (zero-shot under-abstains). The zips were
re-downloaded this session (persistence had been off) and matched the pinned sha256s exactly.

**Prefetching.** `run_predictions(prefetch_workers=N)` prepares the next batches (image load +
processor) in worker threads while the GPU runs the current one. Output is byte-identical to
N=0 (tested with the fake backend and the real SmolVLM/LFM processors). Thread safety: each
worker thread uses its own deep copy of the processor (fast tokenizers can raise "Already
borrowed" when shared), and ZipImages keeps one zip handle per thread. Early stops shut the pool
down at once; a worker's exception surfaces at its own batch. Default `--prefetch-workers 2`.

**Correction.** I told the user the stand-in runs' 0.55 s/question was mostly CPU image
preprocessing and that prefetching could save up to ~4 GPU h. Measuring the real SmolVLM
processor locally (M2, 1296x968 images) gave 0.078 s/image serial and only a 1.1x speedup with 3
threads: the fast image processor already uses several cores through PyTorch, so Python threads
add little. The 0.55 s was more likely fixed warm-up cost (CUDA init, first-call overheads)
spread over only 16 questions. Prefetching still overlaps preprocessing with GPU work, but the
realistic saving is at most the preprocessing time (~0.1-0.3 s/question on Kaggle's slower CPU),
not 0.55. The size of the effect is measured on Kaggle with an A/B smoke run before the dev
runs, rather than assumed.

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
