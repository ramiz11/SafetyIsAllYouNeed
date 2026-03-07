
# LLM4POI Baseline (pre‑Llama‑3.1)

This baseline mirrors the LLM4POI setup (trajectory prompting + LoRA fine‑tuning) **without safety injection** and **without Llama‑3.1**.
It follows the **limited‑history variant described in our paper** (historical trajectories from the same user only, **no key‑query similarity**).

## Run

```bash
python baselines/llm4poi/run_llm4poi_baseline.py \
  --mode both \
  --dataset NYC \
  --traj-len 20 \
  --crime-radius 500 \
  --crime-time-weeks 4 \
  --base-dir /absolute/path/to/SafetyIsAllYouNeed \
  --model-name Llama-2-7b-longlora-32k
```

> **Important:** `Llama-2-7b-longlora-32k` is a placeholder. Set `--model-name` to the exact HF checkpoint or local path you want reviewers to use.
> This baseline does **not** implement the key‑query similarity module from LLM4POI.

Optional:
- `--max-hist-trajs N` limits same‑user historical trajectories (0 = all).
