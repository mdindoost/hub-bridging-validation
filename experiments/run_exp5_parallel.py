#!/usr/bin/env python3
"""
Parallel Experiment 5 Runner
============================

Runs Experiment 5 on multiple networks in parallel, each in its own process.
Each network writes to its own CSV file, then merges at the end.

Usage:
    python run_exp5_parallel.py --workers 4
    python run_exp5_parallel.py --workers 8 --quick
"""

import argparse
import csv
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_networks_for_experiment_5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_network(net_name: str, args_dict: dict) -> dict:
    """
    Run experiment on a single network in a subprocess.

    Returns dict with network name and result status.
    """
    cmd = [
        sys.executable,
        "experiments/run_realism_validation.py",
        "--exp5",
        "--extended",
        "--networks", net_name,
        "--data-dir", args_dict["data_dir"],
        "--output", args_dict["output"],
        "--seed", str(args_dict["seed"]),
    ]

    if args_dict.get("quick"):
        cmd.append("--quick")

    if args_dict.get("n_samples"):
        cmd.extend(["--n-samples", str(args_dict["n_samples"])])

    logger.info(f"Starting: {net_name}")
    start_time = time.time()

    # Create log file for this network
    log_dir = Path(args_dict["output"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{net_name}.log"

    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=7200,  # 2 hour timeout per network
            )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✓ Completed: {net_name} ({elapsed:.1f}s) - log: {log_file}")
            return {"network": net_name, "status": "success", "time": elapsed, "log": str(log_file)}
        else:
            # Read last part of log for error info
            with open(log_file, 'r') as f:
                log_content = f.read()
            last_lines = log_content[-1000:] if len(log_content) > 1000 else log_content
            logger.error(f"✗ Failed: {net_name} - see {log_file}")
            return {"network": net_name, "status": "failed", "log": str(log_file), "error": last_lines}

    except subprocess.TimeoutExpired:
        logger.error(f"✗ Timeout: {net_name} - see {log_file}")
        return {"network": net_name, "status": "timeout", "log": str(log_file)}
    except Exception as e:
        logger.error(f"✗ Error: {net_name} - {e}")
        return {"network": net_name, "status": "error", "error": str(e)}


def merge_csv_files(output_dir: Path, timestamp: str) -> Path:
    """Merge all exp5_extended CSV files into one combined file."""
    csv_files = sorted(output_dir.glob("exp5_extended_*.csv"))

    # Exclude summary files and combined files
    csv_files = [f for f in csv_files if '_summary' not in f.name and 'COMBINED' not in f.name]

    if not csv_files:
        logger.warning("No CSV files found to merge")
        return None

    merged_path = output_dir / f"exp5_extended_COMBINED_{timestamp}.csv"

    header_written = False
    rows_written = 0

    with open(merged_path, 'w', newline='') as outfile:
        writer = None

        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    header = next(reader)

                    if not header_written:
                        writer = csv.writer(outfile)
                        writer.writerow(header)
                        header_written = True

                    for row in reader:
                        if row and row[0]:  # Skip empty rows
                            writer.writerow(row)
                            rows_written += 1
            except Exception as e:
                logger.warning(f"Could not read {csv_file}: {e}")

    logger.info(f"Merged {len(csv_files)} files into {merged_path} ({rows_written} networks)")
    return merged_path


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 5 in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--data-dir", type=str, default="data/real_networks", help="Network data directory")
    parser.add_argument("--output", type=str, default="data/results/realism", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer samples)")
    parser.add_argument("--n-samples", type=int, default=None, help="Samples per network")
    parser.add_argument("--networks", nargs="+", default=None, help="Specific networks to process")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load networks to get the list
    logger.info(f"Loading networks from {args.data_dir}...")
    networks = load_networks_for_experiment_5(
        data_dir=args.data_dir,
        min_nodes=100 if not args.quick else 30,
        max_nodes=100000 if not args.quick else 10000,
    )

    if args.networks:
        networks = {k: v for k, v in networks.items() if any(n in k for n in args.networks)}

    # Sort by size (smallest first for faster initial results)
    network_names = sorted(
        networks.keys(),
        key=lambda x: networks[x]["G"].number_of_nodes()
    )

    logger.info(f"\n{'=' * 70}")
    logger.info(f"PARALLEL EXPERIMENT 5")
    logger.info(f"{'=' * 70}")
    logger.info(f"Networks: {len(network_names)}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    for name in network_names:
        n = networks[name]["G"].number_of_nodes()
        e = networks[name]["G"].number_of_edges()
        logger.info(f"  {name}: {n} nodes, {e} edges")

    logger.info("")

    # Clear old CSV files (but keep pickles for debugging)
    for old_csv in output_dir.glob("exp5_extended_*.csv"):
        old_csv.unlink()
        logger.info(f"Removed old: {old_csv.name}")

    # Run in parallel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args_dict = {
        "data_dir": args.data_dir,
        "output": args.output,
        "seed": args.seed,
        "quick": args.quick,
        "n_samples": args.n_samples,
    }

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_single_network, name, args_dict): name
            for name in network_names
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    elapsed = time.time() - start_time

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info(f"PARALLEL EXECUTION COMPLETE")
    logger.info(f"{'=' * 70}")

    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    logger.info(f"Successful: {len(success)}/{len(results)}")
    logger.info(f"Total time: {elapsed:.1f}s")
    if success:
        avg_time = sum(r.get("time", 0) for r in success) / len(success)
        logger.info(f"Average time per network: {avg_time:.1f}s")

    if failed:
        logger.info(f"\nFailed networks:")
        for r in failed:
            logger.info(f"  {r['network']}: {r['status']}")

    # Merge CSV files
    logger.info(f"\nMerging CSV files...")
    merged_path = merge_csv_files(output_dir, timestamp)

    if merged_path:
        logger.info(f"\n✓ Combined results: {merged_path}")


if __name__ == "__main__":
    main()
