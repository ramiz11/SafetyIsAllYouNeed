from __future__ import annotations
from configs import preprocessing_config as pc
import argparse
import gc
import json
from pathlib import Path
from text_utils import read_json
from train import select_row

def write_new_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def check_inputs(root, config):
    from eval import load_numeric_splits
    from text_utils import verify_canonical_prompts
    from train import load_and_verify_prompts
    from text_utils import sha256_file

    checked = {}
    for city, profile in config["data_profiles"].items():
        manifest = verify_canonical_prompts(root / profile["path"])
        load_numeric_splits(root, profile)
        for name in ("crime_source", "route_cache", "normalization_stats"):
            artifact = profile["safety_runtime"][name]
            if sha256_file(root / artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Safety input changed: {artifact['path']}")
        checked[city] = manifest["source_audit"]
    for row in config["rows"]:
        load_and_verify_prompts(root, config, row, verify_manifest=False)
    return {"status": "inputs_verified", "cities": checked}


def run_rows(root, config_path, config, rows, output_dir):
    from train import run_training
    from eval import run_inference

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        row_dir = output / row["row_key"].replace("|", "_")
        trained = run_training(repo_root=root, config_path=config_path, row_key=row["row_key"],
                               output_dir=row_dir / "training")
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        inferred = run_inference(repo_root=root, config_path=config_path, row_key=row["row_key"],
                                 adapter_path=trained["selected_checkpoint"], output_dir=row_dir / "evaluation")
        results.append(read_json(inferred["result"]))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {
        "evaluation_contract": config["evaluation_contract"],
        "configuration_summary": config["configuration_summary"],
        "rows": results,
    }
    write_new_json(output / "evaluation_results.json", report)
    return report


def main(argv=None, *, command=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    if command is None:
        parser.add_argument("command", choices=("check", "prompts", "train", "infer", "evaluate", "run"))
    else:
        parser.set_defaults(command=command)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--row",
        help="CITY|our_method, or all to run the proposed method for both cities",
    )
    parser.add_argument("--adapter", type=Path, help="Your locally trained checkpoint directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--safety", type=Path)
    args = parser.parse_args(argv)
    if args.command in ("train", "infer", "evaluate", "run") and (not args.row or not args.output_dir):
        parser.error(f"{args.command} requires --row and --output-dir")
    if args.command == "infer" and not args.adapter:
        parser.error("infer requires --adapter from your own training run")
    if args.command == "evaluate" and (not args.predictions or not args.safety):
        parser.error("evaluate requires --predictions and --safety from the same run")
    if args.resume_from_checkpoint and args.command != "train":
        parser.error("--resume-from-checkpoint is only supported by train")
    root = args.repo_root.resolve()
    config_path = root / args.config if args.config else None
    config = pc.load_model_config(config_path)
    if args.command == "check":
        result = check_inputs(root, config)
    elif args.command == "prompts":
        from text_utils import write_canonical_prompts
        result = {city: write_canonical_prompts(root / profile["path"])
                  for city, profile in config["data_profiles"].items()}
    elif args.command == "train":
        from train import run_training
        result = run_training(repo_root=root, config_path=config_path, row_key=args.row,
                              output_dir=args.output_dir, resume_from_checkpoint=args.resume_from_checkpoint)
    elif args.command == "infer":
        from eval import run_inference
        result = run_inference(repo_root=root, config_path=config_path, row_key=args.row,
                               adapter_path=args.adapter, output_dir=args.output_dir)
    elif args.command == "evaluate":
        from eval import evaluate_run
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"Evaluation output directory must be empty: {args.output_dir}")
        weights, result = evaluate_run(root, config, args.row, args.predictions, args.safety)
        write_new_json(args.output_dir / "weights.json", weights)
        write_new_json(args.output_dir / "result.json", result)
    else:
        rows = config["rows"] if args.row == "all" else [select_row(config, args.row)]
        result = run_rows(root, config_path, config, rows, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def run_preprocessing_grid(dataset="CHICAGO", base_dir="."):
    """Run the configured preprocessing parameter sweep explicitly."""
    import run_preprocessing
    for traj_len in (25, 20, 150, 10):
        for radius in (1000, 750, 500, 250):
            for time_window in (4, 3, 2, 1):
                run_preprocessing.main(dataset=dataset, traj_len=traj_len,
                    crime_radius=radius, crime_time_weeks=time_window, base_dir=base_dir)

if __name__ == "__main__":
    main()
