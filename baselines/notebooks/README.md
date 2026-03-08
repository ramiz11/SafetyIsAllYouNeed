## Notebooks

These notebooks were originally created for Google Colab and are kept as reference only.

Recommended baseline workflow:
1) Run preprocessing in the repo
2) Export baseline inputs via `scripts/export_baseline_inputs.py`
3) Use `scripts/run_*.sh` to copy exported files into the upstream baseline repos (STAN / GETNext / STHGCN)
4) Train/evaluate baselines using the upstream repos’ instructions

Note: notebook outputs are intentionally stripped to avoid committing stale/failed runs.
