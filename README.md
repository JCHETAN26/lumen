# Lumen

[![CI](https://github.com/JCHETAN26/lumen/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JCHETAN26/lumen/actions/workflows/ci.yml)

An on-device visual assistant for blind and low-vision users. Lumen fine-tunes a small open
vision-language model (SmolVLM2-500M) with LoRA on [VizWiz-VQA](https://vizwiz.org/tasks-and-datasets/vqa/),
teaches it to say when a photo can't answer the question, quantizes it to 4 bits, and runs it
locally on a MacBook Air (M2, 8 GB) with MLX. An accessible React/TypeScript app comes after.

**Status:** Stage 1 (repo, data pipeline, evaluation code). See [`buildplan.md`](buildplan.md).
No results yet: every number will come from a script, with a baseline and hardware recorded.

## Layout

```
configs/pins.yaml   dataset files (size, sha256) and model revisions (commit SHAs)
configs/splits/     committed question-ID lists (dev split, Mac sample)
src/lumen/          eval/ (official VizWiz metrics), data/ (download, annotations, splits, images),
                    zeroshot/ and train/ (Kaggle), ondevice/ (MLX on the Mac)
kaggle/             notebook runner for all downloads, baselines, and training (Tesla T4)
scripts/            Mac scripts (Stage 4: docs/mac.md) and the predictions guard
app/web/            the accessible web app (Stage 5: docs/app.md); src/lumen/app/ is its server
tests/              unit tests; tests/live/ calls real services and skips on usage caps
results/            result JSONs, each with commit, hardware, and library versions
```

## Development

```bash
uv sync                          # core + dev tools
uv run ruff check . && uv run ruff format --check .
uv run pytest -m "not live"      # unit tests (what CI runs)
uv run pytest -m live            # live checks against Hugging Face / Kaggle / VizWiz host
git config core.hooksPath .githooks   # once per clone: blocks committing model predictions
```

Model prediction files never enter git: they can repeat private details from users' photos. They
are stored in a private Hugging Face dataset, and results here keep only aggregate scores plus a
pinned pointer. A pre-commit hook and CI (`scripts/check_no_predictions.py`) enforce this.

GPU work runs on Kaggle: see [`kaggle/README.md`](kaggle/README.md). On-device work (MLX on the
M2 Air) is in [`docs/mac.md`](docs/mac.md); the app is in [`docs/app.md`](docs/app.md).

## Dataset: VizWiz-VQA

Lumen trains and evaluates on VizWiz-VQA, questions asked by blind people about photos they took
themselves, each with 10 crowdsourced answers and a label for whether it can be answered.

- Website and downloads: <https://vizwiz.org/tasks-and-datasets/vqa/>
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Citation:

```bibtex
@inproceedings{gurari2018vizwiz,
  title     = {VizWiz Grand Challenge: Answering Visual Questions from Blind People},
  author    = {Gurari, Danna and Li, Qing and Stangl, Abigale J. and Guo, Anhong and Lin, Chi and
               Grauman, Kristen and Luo, Jiebo and Bigham, Jeffrey P.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018}
}
```

### Evaluation

Lumen's accuracy metric is a line-for-line port of the official VizWiz evaluation code
(`vqaEval.py` from the VizWiz API, sha256 pinned), tested against that code's own outputs.
Answerability uses the official average precision (unanswerable as the positive class) and F1.

Two quirks of the official code are **kept on purpose**, so Lumen's numbers are comparable with
published VizWiz results:

1. **Only the model's answer is normalized.** Punctuation, number words ("two" → "2"), articles,
   and contractions are normalized in the prediction, but the ten human answers are compared as
   written. A prediction of "two" therefore scores 0 against human answers of "two".
2. **The period-stripping regex has a typo.** `(?!<=\d)(\.)(?!\d)` was meant as the lookbehind
   `(?<!\d)`. As written, it removes any period not followed by a digit ("3." → "3", "3.5" kept).

See [`docs/devlog.md`](docs/devlog.md) for the reasoning. The data stage reports how many ground-truth
answers the first quirk makes unmatchable.

**The test split is sealed.** VizWiz's test answers are public, so Lumen enforces its own rule:
all tuning happens on a dev split carved from train, val is the reporting split, and test is
evaluated exactly once. Test answers load only inside that single run, which needs an explicit
`--final-test-evaluation` flag and a clean commit, and records its commit, date, and results in
`results/test/FINAL_TEST_RUN.json`. Once that file exists, test evaluation refuses to run again.

### Respectful use of these photos

These images were taken by blind and low-vision people asking for help in their daily lives.
They show people's homes, medicine, mail, screens, and sometimes their faces or private
information, often without the photographer being able to see what the frame caught. So:

- The images are used only to train and evaluate Lumen. They are never committed to this repo,
  shown in the README, demos, slides, or social posts, or redistributed.
- No example in any write-up quotes a photo's private details (names, addresses, account numbers,
  prescriptions) even when they are visible.
- Error analysis talks about categories of failure (blur, framing, lighting), not individuals.
- The goal is a tool that serves the people who contributed this data. When Lumen is unsure, it
  says so, rather than giving a confident wrong answer to someone who can't check it.

## Models

| Model | Role | License |
|---|---|---|
| [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | fine-tuned and shipped | Apache-2.0 |
| [SmolVLM2-256M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) | zero-shot baseline | Apache-2.0 |
| [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | zero-shot baseline (upper reference) | Apache-2.0 |
| [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) | zero-shot baseline | LFM Open License v1.0 |

Exact revisions are pinned by commit SHA in [`configs/pins.yaml`](configs/pins.yaml).
