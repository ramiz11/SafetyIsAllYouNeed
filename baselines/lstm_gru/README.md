
# LSTM / GRU Baselines

Minimal sequential baselines using the same preprocessed trajectories as the main pipeline.

```bash
python baselines/lstm_gru/run_lstm_gru.py   --model lstm   --mode both   --dataset NYC   --traj-len 20   --crime-radius 500   --crime-time-weeks 4   --base-dir /absolute/path/to/SafetyIsAllYouNeed
```

Use `--model gru` to train the GRU baseline.
