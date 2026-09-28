# Dev log

Key decisions and anything that went wrong, newest first.

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
