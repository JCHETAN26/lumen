# Dev log

Key decisions and anything that went wrong, newest first.

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
