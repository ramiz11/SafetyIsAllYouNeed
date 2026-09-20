"""Compatibility wrapper for the public, self-trained LLM4POI configurations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batch_runner import main as batch_main
from configs import preprocessing_config as pc
from train import select_row


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval", "both"), default="both")
    parser.add_argument("--dataset", choices=("NYC", "CHICAGO"), required=True)
    parser.add_argument("--variant", choices=("llm4poi_31", "llm4poi_original"), required=True)
    parser.add_argument("--base-dir", default=str(REPO_ROOT))
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint")
    # Existing explicit settings are accepted only when they match the chosen recipe.
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base-precision", choices=("bnb_default", "nf4"))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    row_key = f"{args.dataset}|{args.variant}"
    config = pc.load_model_config(Path(args.base_dir) / args.config if args.config else None)
    row = select_row(config, row_key)
    for name in ("seed", "base_precision"):
        if getattr(args, name) is not None and getattr(args, name) != row[name]:
            parser.error(f"{name} differs from the selected recipe; use an explicit --config")
    if args.resume_from_checkpoint and args.mode != "train":
        parser.error("Resume a training run with --mode train")
    row_dir = args.run_dir / row_key.replace("|", "_")
    command = {"train": "train", "eval": "infer", "both": "run"}[args.mode]
    forwarded = [command, "--repo-root", args.base_dir, "--row", row_key]
    if args.config:
        forwarded += ["--config", args.config]
    if args.mode == "both":
        if args.adapter or args.output_dir:
            parser.error("--mode both trains its own adapter and uses --run-dir")
        forwarded += ["--output-dir", str(args.run_dir)]
    elif args.mode == "train":
        forwarded += ["--output-dir", str(row_dir / "training")]
        if args.resume_from_checkpoint:
            forwarded += ["--resume-from-checkpoint", args.resume_from_checkpoint]
    else:
        adapter = args.adapter or row_dir / "training" / "checkpoints" / f"checkpoint-{row['checkpoint']['step']}"
        forwarded += ["--adapter", str(adapter), "--output-dir", str(args.output_dir or row_dir / "evaluation")]
    return batch_main(forwarded)


if __name__ == "__main__":
    main()
