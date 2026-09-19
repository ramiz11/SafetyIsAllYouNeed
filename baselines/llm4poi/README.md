# LLM4POI baselines

Both no-Safety variants use the shared [public reproduction pipeline](../../REPRODUCING.md).
Users train their own LoRA adapters; no trained checkpoint download is supplied.

- `llm4poi_original`: the configured Llama-2/LongLoRA base model.
- `llm4poi_31`: Llama-3.1-8B-Instruct.

Both receive the 19 observed check-ins, including row-level coordinates, with
the 20th POI as the training answer. Neither receives transition Safety in the
prompt or uses the key-query similarity module. See the central configuration
for per-row precision, seeds, checkpoint steps and inference settings.

## Train, infer and evaluate

```bash
python baselines/llm4poi/run_llm4poi_baseline.py \
  --mode both --dataset CHICAGO --variant llm4poi_original \
  --run-dir runs/chicago_original
```

Switch to `--variant llm4poi_31` for the Llama-3.1 baseline. Use `--mode train`
for training only, then `--mode eval` with the same run directory to load the
configured checkpoint from that training run. `--adapter` can specify another
locally trained checkpoint with its matching training manifest.

Run directories must be new for `--mode both`. To resume interrupted training,
use `--mode train --resume-from-checkpoint /path/to/checkpoint-N`.
The wrapper delegates to the main reproduction runner.
