"""
cli.py

Unified Developer & Production CLI for the DeepCell Bioimaging Platform.
Supports pipeline execution, ground-truth benchmark evaluation (PQ / AJI),
throughput profiling, and REST API microservice serving.
"""

import os
import sys
import argparse
import subprocess
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gpu_utils import print_hardware_summary
from src.evaluation_metrics import compute_panoptic_quality, compute_aggregated_jaccard_index


def print_banner():
    banner = r"""
  ____  _____ _____ ____   ____ _____ _     _     
 |  _ \| ____| ____|  _ \ / ___| ____| |   | |    
 | | | |  _| |  _| | |_) | |   |  _| | |   | |    
 | |_| | |___| |___|  __/| |___| |___| |___| |___ 
 |____/|_____|_____|_|    \____|_____|_____|_____|
 High-Throughput Bioimaging & Spatial Biology Platform
    """
    print("\033[96m" + banner + "\033[0m")


def cmd_run(args):
    """Executes the complete segmentation & spatial pipeline."""
    from run_pipeline import main as run_main
    sys.argv = [
        "run_pipeline.py",
        "--config", args.config,
        "--input-dir", args.input_dir,
        "--output-dir", args.output_dir,
        "--sample-name", args.sample_name
    ]
    if args.arcsinh:
        sys.argv.append("--arcsinh")
    run_main()


def cmd_evaluate(args):
    """Computes Panoptic Quality (PQ) and Aggregated Jaccard Index (AJI)."""
    import tifffile
    print("\n" + "=" * 65)
    print(" QUANTITATIVE SEGMENTATION EVALUATION BENCHMARK")
    print("=" * 65)
    print(f" Ground Truth Mask : {args.ground_truth}")
    print(f" Predicted Mask    : {args.prediction}")

    gt = tifffile.imread(args.ground_truth)
    pred = tifffile.imread(args.prediction)

    pq_res = compute_panoptic_quality(gt, pred, iou_threshold=args.iou_threshold)
    aji_res = compute_aggregated_jaccard_index(gt, pred)

    print("-" * 65)
    print(f" Panoptic Quality (PQ)       : {pq_res['pq']:.4f}")
    print(f" Segmentation Quality (SQ)   : {pq_res['sq']:.4f} (Average IoU of matches)")
    print(f" Recognition Quality (RQ)    : {pq_res['rq']:.4f} (F1 Detection Score)")
    print(f" Aggregated Jaccard Index    : {aji_res:.4f}")
    print(f" True Positives (IoU >= {args.iou_threshold}): {pq_res['tp']}")
    print(f" False Positives             : {pq_res['fp']}")
    print(f" False Negatives             : {pq_res['fn']}")
    print("=" * 65 + "\n")


def cmd_serve(args):
    """Launches the FastAPI microservice."""
    import uvicorn
    print(f"\n[INFO] Starting DeepCell REST API on http://{args.host}:{args.port}")
    print(f"[INFO] Interactive OpenAPI Documentation: http://{args.host}:{args.port}/docs\n")
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_synthetic(args):
    """Generates synthetic test data."""
    from sample_data.generate_synthetic_data import main as syn_main
    syn_main()


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="DeepCell Bioimaging & Spatial Biology CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # run command
    p_run = subparsers.add_parser("run", help="Execute segmentation & spatial pipeline")
    p_run.add_argument("--config", default="config/pipeline_config.yaml")
    p_run.add_argument("--input-dir", default="data/tiles_export")
    p_run.add_argument("--output-dir", default="data/segmentation_output")
    p_run.add_argument("--sample-name", default="Sample_01")
    p_run.add_argument("--arcsinh", action="store_true")

    # evaluate command
    p_eval = subparsers.add_parser("evaluate", help="Compute PQ, SQ, RQ, and AJI against ground truth")
    p_eval.add_argument("--ground-truth", required=True, help="Path to ground truth labeled mask TIFF")
    p_eval.add_argument("--prediction", required=True, help="Path to predicted labeled mask TIFF")
    p_eval.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for true positive match")

    # serve command
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI REST microservice")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    # synthetic command
    p_syn = subparsers.add_parser("synthetic", help="Generate synthetic multi-channel test data")

    # hardware command
    p_hw = subparsers.add_parser("hardware", help="Display CUDA GPU accelerator status")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "synthetic":
        cmd_synthetic(args)
    elif args.command == "hardware":
        print_hardware_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

