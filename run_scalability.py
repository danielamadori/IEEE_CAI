#!/usr/bin/env python3
"""
Run init_aeon_univariate.py for multiple datasets and test worker scalability.

For each dataset, this script initializes Redis and then starts workers using
32 and 16 processes (configurable via --workers). Workers run for a fixed
amount of time or until you press Enter.
"""

from __future__ import annotations

import csv
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Configuration variables
datasets = ["Wine", ] 
    # "MiddlePhalanx OutlineCorrect", "SonyAIBO RobotSurface1",
    #             "Beetle Fly", "TwoLead ECG", "Lightning 2", "Face Four", "ToeSegmentation 2",
    #              "ECG200", "ItalyPower Demand", "Meat", "SonyAIBO RobotSurface2", "Coffee" 
                #"Bird Chicken" , "Gun Point", "CinC ECGTorso", "Mote Strain" ]  # List of dataset names to initialize
workers = [16]  # Worker counts to test
class_labels = {
    # Optional manual overrides per dataset, e.g. "Wine": "1"
}
auto_class_label_strategy = "first"  # "first", "majority", or "none"
redis_port = 6379  # Redis/KeyDB port to pass to init script
init_args = ""  # Extra arguments for init_aeon_univariate.py (quoted string)
worker_script = "worker_cache_logged.py"  # Worker script to run
worker_args = ""  # Extra arguments for workers (passed via launch_workers.py --args)
run_seconds = 0  # Seconds to keep workers running; 0 waits for Enter before stopping
skip_initial_stop = False  # Do not stop existing workers before starting a run
dataset_info_file = 'dataset_info.csv'

def _flag_present(tokens: list[str], flag: str) -> bool:
    for token in tokens:
        if token == flag or token.startswith(flag + "="):
            return True
    return False


def _run_cmd(cmd: list[str], cwd: Path, check: bool = True) -> int:
    pretty = " ".join(cmd)
    print(f"\n[CMD] {pretty}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def _stop_workers(python_exe: str, cwd: Path) -> None:
    _run_cmd([python_exe, "launch_workers.py", "stop"], cwd=cwd, check=False)


def _auto_class_label(dataset: str, strategy: str) -> str:
    try:
        from aeon.datasets import load_classification
    except Exception as exc:
        raise RuntimeError(
            "Auto class label requires aeon. Install dependencies or set class_labels."
        ) from exc

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "Auto class label requires numpy. Install dependencies or set class_labels."
        ) from exc

    try:
        _, y_train = load_classification(dataset, split="train")
        _, y_test = load_classification(dataset, split="test")
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset '{dataset}' for class labels: {exc}") from exc

    labels = np.concatenate([y_train, y_test])
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size == 0:
        raise RuntimeError(f"No class labels found for dataset '{dataset}'")

    if strategy == "majority":
        chosen = classes[counts.argmax()]
    else:
        chosen = classes[0]

    return str(chosen)


def _load_dataset_info(csv_file: Path) -> dict[str, dict]:
    """Load dataset information from CSV file"""
    dataset_info = {}
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset_name = row['Dataset']
                dataset_info[dataset_name] = {
                    'train_size': int(row['Train Size']),
                    'test_size': int(row['Test Size']),
                    'length': int(row['Length']),
                    'num_classes': int(row['No. of Classes']),
                    'type': row['Type']
                }
    except FileNotFoundError:
        print(f"[WARNING] Dataset info file not found: {csv_file}")
    except Exception as e:
        print(f"[WARNING] Error reading dataset info: {e}")
    return dataset_info


def main() -> int:

    repo_root = Path(__file__).resolve().parent
    init_script = repo_root / "init_aeon_univariate.py"
    launch_script = repo_root / "launch_workers.py"
    dataset_info_path = repo_root / dataset_info_file

    if not init_script.exists():
        print(f"[ERROR] Missing init script: {init_script}")
        return 1
    if not launch_script.exists():
        print(f"[ERROR] Missing worker launcher: {launch_script}")
        return 1

    # Load dataset information
    dataset_info = _load_dataset_info(dataset_info_path)

    init_tokens = shlex.split(init_args, posix=os.name != "nt") if init_args else []
    init_has_class_label = _flag_present(init_tokens, "--class-label")

    if run_seconds < 0:
        print("[ERROR] run_seconds must be >= 0")
        return 1

    python_exe = sys.executable

    try:
        if not skip_initial_stop:
            _stop_workers(python_exe, repo_root)

        for dataset in datasets:
            # Get dataset info
            info = dataset_info.get(dataset)
            if info:
                num_classes = info['num_classes']
                print(f"\n[INFO] Dataset '{dataset}': {num_classes} classes, Type: {info['type']}")
            else:
                print(f"\n[WARNING] No info found for dataset '{dataset}' in {dataset_info_file}")
                num_classes = None

            class_label = class_labels.get(dataset)
            if class_label is None and not init_has_class_label:
                if auto_class_label_strategy == "none":
                    print(f"[ERROR] Missing class label for dataset '{dataset}'.")
                    print("        Set class_labels or add --class-label to init_args.")
                    return 1
                class_label = _auto_class_label(dataset, auto_class_label_strategy)
                print(
                    f"[AUTO] Using class label '{class_label}' for '{dataset}' "
                    f"(strategy={auto_class_label_strategy})"
                )
            elif class_label is not None and init_has_class_label:
                print(
                    "[WARNING] class_labels is set but init_args already includes --class-label. "
                    "init_args will take precedence."
                )
            
            for worker_count in workers:
                print("\n" + "=" * 80)
                print(f"[RUN] Dataset={dataset} | Workers={worker_count}")
                if num_classes:
                    print(f"      Classes={num_classes}")
                if class_label and not init_has_class_label:
                    print(f"      Class label={class_label}")
                print("=" * 80)

                if not skip_initial_stop:
                    _stop_workers(python_exe, repo_root)

                # Initialize dataset - class label will be taken from the dataset itself
                init_cmd = [python_exe, str(init_script), dataset]
                if class_label is not None and not init_has_class_label:
                    init_cmd += ["--class-label", str(class_label)]
                if redis_port is not None and not _flag_present(init_tokens, "--redis-port"):
                    init_cmd += ["--redis-port", str(redis_port)]
                init_cmd += init_tokens

                _run_cmd(init_cmd, cwd=repo_root, check=True)

                start_cmd = [
                    python_exe,
                    str(launch_script),
                    "start-legacy",
                    str(worker_count),
                    "--worker",
                    worker_script,
                ]
                if worker_args:
                    start_cmd += ["--args", worker_args]

                _run_cmd(start_cmd, cwd=repo_root, check=True)

                if run_seconds == 0:
                    input("\nWorkers running. Press Enter to stop and continue... ")
                else:
                    print(f"\nSleeping for {run_seconds}s...")
                    time.sleep(run_seconds)

                _stop_workers(python_exe, repo_root)

        print("\n[OK] Scalability runs complete.")
        return 0

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping workers...")
        _stop_workers(python_exe, repo_root)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Command failed with exit code {exc.returncode}:")
        print(f"        {' '.join(exc.cmd)}")
        _stop_workers(python_exe, repo_root)
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
