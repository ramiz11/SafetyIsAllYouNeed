
# LSTM / GRU Baselines

Minimal sequential baselines using the same preprocessed trajectories as the main pipeline.

```bash
python baselines/lstm_gru/run_lstm_gru.py   --model lstm   --mode both   --dataset CHICAGO   --traj-len 20   --crime-radius 1000   --crime-time-weeks 3   --base-dir /absolute/path/to/SafetyIsAllYouNeed
```

Use `--model gru` to train the GRU baseline.
