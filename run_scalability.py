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
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from redis_backup import (
    DEFAULT_REDIS_DATABASES,
    create_multi_database_backup,
    save_multi_database_backup_to_directory,
    build_redis_client,
)

# Configuration variables
datasets = ["Wine", "MiddlePhalanxOutlineCorrect", "SonyAIBORobotSurface1",
                "BeetleFly", "TwoLeadECG", "Lightning2", "FaceFour", "ToeSegmentation2",
                 "ECG200", "ItalyPowerDemand", "Meat", "SonyAIBORobotSurface2", "Coffee", 
                "BirdChicken" , "GunPoint", "CinCECGTorso", "MoteStrain" ]  # List of dataset names to initialize
workers = [16]  # Worker counts to test
class_labels = {
    # Optional manual overrides per dataset, e.g. "Wine": "1"
}

redis_port = 6379  # Redis/KeyDB port to pass to init script
init_args = ""  # Extra arguments for init_aeon_univariate.py (quoted string)
worker_script = "worker_cache_logged.py"  # Worker script to run
worker_args = ""  # Extra arguments for workers (passed via launch_workers.py --args)
skip_initial_stop = True  # Do not stop existing workers before starting a run
dataset_info_file = 'dataset_info.csv'
redis_config: Dict[str, Any] = {
    "host": "127.0.0.1",
    "port": redis_port,
    "db": 0,
}
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

def _auto_class_label(dataset: str) -> str:
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
    return np.unique(labels)


def clean_up(dataset, worker_count: int, repo_root: Path, redis_config, class_label) -> None:
    
        # Save Redis backup after experiment
        print(f"[BACKUP] Saving Redis data for {dataset} with {worker_count} workers...")
        
        
        backup_dir = repo_root / "results" / "checkpoints" / dataset / class_label / f"{dataset}_{worker_count}_backup"
        try:
            backup_payloads = create_multi_database_backup(
                redis_config,
                databases=DEFAULT_REDIS_DATABASES,
                scan_count=1000,
            )
            save_multi_database_backup_to_directory(
                backup_payloads,
                backup_dir,
                file_prefix=f"redis_backup_{dataset}_{worker_count}",
            )
            total_keys = sum(payload["metadata"]["key_count"] for payload in backup_payloads.values())
            print(f"[BACKUP] Saved {total_keys} keys across {len(backup_payloads)} databases to {backup_dir}")
        except Exception as e:
            print(f"[WARNING] Backup failed: {e}")
        
        # Clean databases before next experiment
        print(f"[CLEANUP] Flushing Redis databases before next experiment...")
        try:
            for db_num in DEFAULT_REDIS_DATABASES:
                db_config = dict(redis_config)
                db_config["db"] = db_num
                client = build_redis_client(db_config)
                client.flushdb()
            print(f"[CLEANUP] Flushed {len(DEFAULT_REDIS_DATABASES)} databases")
        except Exception as e:
            print(f"[WARNING] Cleanup failed: {e}")
def _parse_status_summary(output: str) -> tuple[int, int] | None:
    status_summary_re = re.compile(r"Summary:\s*(\d+)\s+active,\s*(\d+)\s+inactive", re.IGNORECASE)
    match = status_summary_re.search(output)
    if match:
        return int(match.group(1)), int(match.group(2))
    if "[EMPTY]" in output and "No workers" in output:
        return 0, 0
    return None

def _get_workers_status(cmd, repo_root) -> tuple[int, int] | None:
    
    result = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    print(f"[DEBUG] Status command exited with code {result.returncode}, result: {result}")
    output = (result.stdout or "") + (result.stderr or "")
    print(f"[DEBUG] Status command output:\n{output}")
    if result.returncode != 0:
        print(f"[EXPERIMENT] Warning: status command failed (exit {result.returncode}).")
        return None
    summary = _parse_status_summary(output)
    if summary is None:
        print("[EXPERIMENT] Warning: unable to parse worker status output.")
    return summary

def wait_for_workers(
    cmd,
    repo_root,
    poll_seconds: float = 12.0,
    initial_delay_seconds: float = 30.0,
) -> bool | int:
    """
    Poll `launch_workers.py status` and wait for all workers to exit.
    Returns True if all workers finished, or the active worker count if requested.
    """
    # Give launch_workers a moment to register workers
    time.sleep(initial_delay_seconds)

    print("[EXPERIMENT] Waiting for workers to finish...")

    while True:
        status = _get_workers_status(cmd, repo_root)
        if status is None:
            print("[EXPERIMENT] Worker status: unknown")
            time.sleep(poll_seconds)
            continue

        active_count, inactive_count = status
        print(f"[EXPERIMENT] Worker status: {active_count} active, {inactive_count} inactive")
        print(f"[EXPERIMENT] {active_count} active workers remaining...")

        if active_count == 0:
            print("[EXPERIMENT] All workers exited.")
            return True

        time.sleep(poll_seconds)

def main() -> int:

    repo_root = Path(__file__).resolve().parent
    init_script = repo_root / "init_aeon_univariate.py"
    launch_script = repo_root / "launch_workers.py"

    if not init_script.exists():
        print(f"[ERROR] Missing init script: {init_script}")
        return 1
    if not launch_script.exists():
        print(f"[ERROR] Missing worker launcher: {launch_script}")
        return 1

    init_tokens = shlex.split(init_args, posix=os.name != "nt") if init_args else []
    init_has_class_label = _flag_present(init_tokens, "--class-label")

    python_exe = sys.executable 

    try:
        for worker_count in workers:
            for dataset in datasets:
                class_labels = _auto_class_label(dataset)                
                for class_label in class_labels:

                    print("\n" + "=" * 80)
                    print(f"[RUN] Dataset={dataset} | Workers={worker_count}")
                    if class_label and not init_has_class_label:
                        print(f"      Class label={class_label}")
                    print("=" * 80)

                    # Initialize dataset - class label will be taken from the dataset itself
                    init_cmd = [python_exe, str(init_script), dataset]
                    if class_label is not None and not init_has_class_label:
                        init_cmd += ["--class-label", str(class_label)]
                    if redis_port is not None and not _flag_present(init_tokens, "--redis-port"):
                        init_cmd += ["--redis-port", str(redis_port)]
                    init_cmd += init_tokens
                    file_log = f'results/checkpoints/{dataset}/{class_label}/{dataset}_{worker_count}_log.txt'
                    # file che abbia la lista dei contenuti delle reason da prendere direttamente da redis
                    # 
                    os.makedirs(os.path.dirname(file_log), exist_ok=True)
                    with open(file_log, 'w') as f:
                        f.write(50*'=' + '\n')
                        f.write('Starting experiment with parameters:\n')
                        f.write('dataset: {}\nnum workers: {}\n'.format(dataset, worker_count))
                        f.write(50*'=' + '\n')
                    # launch init script saving data to Redis
                    _run_cmd(init_cmd, cwd=repo_root, check=True)

                    start_cmd = [
                        python_exe,
                        str(launch_script),
                        "start",
                    ]
                    # start_cmd = [
                    #     python_exe,
                    #     str(launch_script),
                    #     "start-legacy",
                    #     str(worker_count),
                    #     "--worker",
                    #     worker_script,
                    # ]
                    # if worker_args:
                    #     start_cmd += ["--args", worker_args]
                    # # launch workers
                    _run_cmd(start_cmd, cwd=repo_root, check=True)

                    # 6. Wait for Completion
                    print(f"[EXPERIMENT] Waiting for solution...")
                    start_wait = time.time()
                    cmd = [
                        python_exe,
                        str(launch_script),
                        "status",
                    ]
                    wait_for_workers(cmd, repo_root, poll_seconds=12.0)
                    duration = time.time() - start_wait
                    print(f"[EXPERIMENT] Iteration {worker_count} for dataset {dataset} class {class_label} completed in {duration:.2f} seconds.")
                    start_cmd = [
                        python_exe,
                        str(launch_script),
                        "stop",
                    ]
                    _run_cmd(start_cmd, cwd=repo_root, check=True)
                    # remove file work
                    with open(file_log, 'w') as f:
                        f.write('Experiment completed in {:.2f} seconds.\n Ended at {}\n'.format(duration, time.strftime("%Y-%m-%d %H:%M:%S")))
                        f.write(50*'=' + '\n')
                    clean_up(dataset=dataset, worker_count=worker_count, repo_root=repo_root, redis_config=redis_config, class_label=class_label)
                    with open(file_log, 'a') as f:
                        f.write('Backup and cleanup done for dataset {} with {} workers.\n'.format(dataset, worker_count))
                        f.write(50*'=' + '\n')
        print("\n[OK] Scalability runs complete.")
        return 0

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping workers...")
        _run_cmd([python_exe, "launch_workers.py", "stop"], cwd=repo_root, check=False)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Command failed with exit code {exc.returncode}:")
        print(f"        {' '.join(exc.cmd)}")
        _run_cmd([python_exe, "launch_workers.py", "stop"], cwd=repo_root, check=False)
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
