# Stage 4 on the Mac: quantization and on-device benchmarks

Everything here runs on the MacBook Air (M2, 8 GB). Nothing is downloaded except the pinned base
model and the pinned adapter, and nothing trains. One heavy job at a time: the Air is fanless.

## Before you start

1. The 300-question Mac sample is extracted to `data/mac_sample/` (from Kaggle's `mac_sample`
   archive; its sha256 is in `results/mac_sample/`). Every image is re-checked against the
   archive manifest on each run.
2. The chosen adapter is pinned. On any machine with the committed train results:
   `uv run python kaggle/run_stages.py pin_adapter --run <run> --epoch <k>` prints the
   `adapters:` entry (Hub commit + every file's sha256); paste it into `configs/pins.yaml` and
   commit.
3. `HF_TOKEN` with read access to the private adapter repo (for the adapter and, optionally, the
   Kaggle predictions used as a reference).
4. Plugged in, lid open, nothing else heavy running.

## Steps

```bash
G="uv run --group gpu --group mlx python"
$G scripts/mac_prepare.py --adapter lr2e-4-e1          # merge (CPU, fp32 -> fp16) + MLX fp16/8bit/4bit
$G scripts/mac_bench.py self-test --model lr2e-4-e1    # must print PASSED
$G scripts/mac_bench.py bench --model lr2e-4-e1 --variant fp16
$G scripts/mac_bench.py bench --model lr2e-4-e1 --variant 8bit
$G scripts/mac_bench.py bench --model lr2e-4-e1 --variant 4bit
$G scripts/mac_bench.py summary --model lr2e-4-e1 --kaggle-result results/finetune/lr2e-4/val-e1.json
```

`--base` / `--model base` does the same for the zero-shot SmolVLM2-500M (same 1,024 px and prompt).

## What each step measures, and why

- **Merge, then convert.** mlx-vlm's own LoRA loader expects its own format, not PEFT's, so the
  adapter is merged into the base with transformers + peft (fp32 on the CPU, saved as fp16) and
  the merged model is converted. mlx-vlm quantizes the language model and connector; the vision
  encoder stays fp16. Sizes on disk and the merged weights' sha256 go to `results/mac/`.
- **Self-test.** On two sample questions: the MLX P(unanswerable) from the KV-cache shortcut must
  match uncached forward passes (within 0.002), and the prompt's token IDs must equal those of the
  transformers processor the Kaggle runs used (same chat template, same image tiling at 1,024 px).
  `bench` refuses to run until the fp16 self-test has passed.
- **Accuracy and AP** on all 300 questions with the official metric, through the same harness as
  Kaggle (same prompt, greedy decoding, 16-token cap, extractor). Predictions stay in the
  gitignored `predictions/mac/`; results get scores and the file's sha256, never answer text.
- **Latency**, the way the app answers: preprocessing + streamed generation, wall-clock, batch 1.
  The first 20 questions, 2 untimed warm-up answers, then 3 timed passes: median and p90 of time to
  first token, total time, and tokens/s, plus peak memory. Each pass's median is recorded, so
  slowing from heat mid-run is visible.
- **Thermal state.** `bench` waits (up to 10 min) for `pmset -g therm` to report no CPU speed
  limit before starting, and records power source and speed limit before, between, and after.
- **Summary.** Each quantized variant against MLX fp16 on the same 300 questions: paired bootstrap
  CIs on the accuracy and AP differences, and how often the answers agree. Kaggle fp16 on the same
  questions is a reference only: mlx-vlm's SmolVLM processor resizes with PIL where Kaggle used the
  torchvision path, so small differences there are the port, not quantization.
