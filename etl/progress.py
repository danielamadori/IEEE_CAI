"""
Utilities for monitoring multi-process ICF computations and coordinating cache writes.

The notebook uses these helpers to keep track of the progress of each reason type
while batches are processed in parallel, and to make cache writes asynchronous in
order to avoid lock contention or deadlocks without sacrificing parallel throughput.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional, Any, List
from multiprocessing import get_context
from pathlib import Path
import time


@dataclass
class _ReasonProgress:
    """Internal structure that tracks progress state for a single reason type."""

    total_batches: int
    total_icfs: int
    start_time: float
    completed_batches: int = 0
    processed_icfs: int = 0
    done: bool = False

    def as_dict(self) -> Dict[str, Any]:
        elapsed = max(time.time() - self.start_time, 1e-9)
        pct = (self.processed_icfs / self.total_icfs * 100.0) if self.total_icfs else 0.0
        rate = self.processed_icfs / elapsed
        return {
            "batches": f"{self.completed_batches}/{self.total_batches}",
            "icfs": f"{self.processed_icfs}/{self.total_icfs}",
            "pct": pct,
            "rate": rate,
            "status": "done" if self.done else "running",
            "elapsed": elapsed,
        }


class ICFProgressMonitor:
    """
    Text-based progress monitor for ICF processing.

    Keeps a snapshot of how many batches/ICFs have been processed for each reason
    type and periodically prints a compact summary. Designed to be lock-safe and
    notebook friendly (plain text output).
    """

    def __init__(self, project_name: Optional[str] = None,
                 refresh_interval: float = 2.0,
                 enabled: bool = True):
        self.project_name = project_name or "ICF Project"
        self.refresh_interval = refresh_interval
        self.enabled = enabled
        self._state: Dict[str, _ReasonProgress] = {}
        self._lock = Lock()
        self._last_render = 0.0

    def start_reason(self, reason_type: str, total_batches: int, total_icfs: int):
        if not self.enabled:
            return
        with self._lock:
            self._state[reason_type] = _ReasonProgress(
                total_batches=total_batches,
                total_icfs=total_icfs,
                start_time=time.time()
            )
            self._render(force=True)

    def batch_completed(self, reason_type: str, icfs_processed: int):
        if not self.enabled:
            return
        with self._lock:
            reason_state = self._state.get(reason_type)
            if not reason_state:
                return
            reason_state.completed_batches += 1
            reason_state.processed_icfs += icfs_processed
            self._render()

    def complete_reason(self, reason_type: str):
        if not self.enabled:
            return
        with self._lock:
            reason_state = self._state.get(reason_type)
            if not reason_state:
                return
            reason_state.done = True
            # Force a final render for this reason type
            self._render(force=True)

    def _render(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_render) < self.refresh_interval:
            return
        self._last_render = now

        header = f"[{self.project_name}] Monitor"
        lines: List[str] = [header]
        for reason_type, state in self._state.items():
            snapshot = state.as_dict()
            lines.append(
                f"  - {reason_type:>12}: "
                f"{snapshot['batches']} batches | "
                f"{snapshot['icfs']} ICFs ({snapshot['pct']:.1f}%) | "
                f"{snapshot['rate']:.1f} ICF/s | "
                f"{snapshot['status']}"
            )
        print("\n".join(lines))


class CacheWriteCoordinator:
    """
    Serialize cache writes inside a dedicated process to keep the main notebook
    thread responsive and avoid throttling the worker pool.
    """

    def __init__(self, cache, zip_path, verbose: bool = False):
        self.verbose = verbose
        self._closed = False
        self._process = None
        self._queue = None

        if cache is None or zip_path is None:
            self._closed = True
            return

        self._zip_path = str(zip_path)
        self._cache_dir = str(getattr(cache, "cache_dir", Path("results/_cache")))

        ctx = get_context("spawn")
        self._queue = ctx.JoinableQueue()
        self._process = ctx.Process(
            target=_cache_writer_process,
            args=(self._queue, self._cache_dir, self._zip_path, self.verbose),
            name="CacheWriteCoordinator",
            daemon=True,
        )
        self._process.start()

    def submit(self, cost_batch, tests_sample_batch: Dict[str, Any],
               reason_type: str, is_first_batch: bool):
        if self._closed or self._process is None:
            return
        payload = {
            "cost_batch": cost_batch,
            "tests_sample_batch": tests_sample_batch,
            "reason_type": reason_type,
            "is_first_batch": is_first_batch
        }
        self._queue.put(payload)

    def flush(self):
        if self._closed or self._process is None:
            return
        self._queue.join()

    def close(self):
        if self._closed or self._process is None:
            return
        self._queue.put(None)
        self._queue.join()
        self._process.join()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _cache_writer_process(queue, cache_dir, zip_path, verbose):
    """Worker loop executed in a separate process."""
    from pathlib import Path as _Path
    from etl.cache import ETLCache  # Local import to avoid circular deps

    cache = ETLCache(_Path(cache_dir))

    while True:
        payload = queue.get()
        if payload is None:
            queue.task_done()
            break
        try:
            cache.save_costs_incremental(
                _Path(zip_path),
                payload["cost_batch"],
                payload["tests_sample_batch"],
                payload["reason_type"],
                is_first_batch=payload["is_first_batch"]
            )
            if verbose:
                batch_size = len(payload["cost_batch"])
                print(f"[CacheWriteCoordinator] Saved {batch_size} cost rows for {payload['reason_type']}")
        finally:
            queue.task_done()
