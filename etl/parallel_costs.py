"""
Parallel cost calculation for reasons analysis - TRUE MULTIPROCESSING
Uses ProcessPoolExecutor for real parallel execution across multiple CPU cores.
Each batch runs in a separate process (no GIL limitation).
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Tuple, Optional, Any
import time

from etl.progress import ICFProgressMonitor, CacheWriteCoordinator


_WORKER_CONTEXT: Optional[Dict[str, Any]] = None


def _init_cost_worker(test_ids, tests_sample, sigmas_all, cost_function, bitmap_to_icf, eu):
    """
    Initializer executed in every worker process so heavy data structures are
    transferred once instead of per-task.
    """
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "test_ids": test_ids,
        "tests_sample": tests_sample,
        "sigmas_all": sigmas_all,
        "cost_function": cost_function,
        "bitmap_to_icf": bitmap_to_icf,
        "eu": eu,
    }


def calculate_cost_for_icf_batch(batch_id: int, reason_type: str, bitmap_strings: List[str]) -> Tuple[int, str, List[Dict], int]:
    """
    Calculate costs for a batch of ICFs across all samples

    This function runs in a SEPARATE PROCESS (true parallelism, no GIL)

    Parameters
    ----------
    batch_id : int
        Sequential identifier for the batch
    reason_type : str
        Reason family being processed
    bitmap_strings : list
        Bitmaps belonging to this batch

    Returns
    -------
    tuple
        (batch_id, reason_type, results_list, icfs_processed)
    """
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context not initialized")

    test_ids = _WORKER_CONTEXT["test_ids"]
    tests_sample = _WORKER_CONTEXT["tests_sample"]
    sigmas_all = _WORKER_CONTEXT["sigmas_all"]
    cost_function = _WORKER_CONTEXT["cost_function"]
    bitmap_to_icf = _WORKER_CONTEXT["bitmap_to_icf"]
    eu = _WORKER_CONTEXT["eu"]

    results = []
    processed_icfs = len(bitmap_strings)
    for bitmap_string in bitmap_strings:
        try:
            icf = bitmap_to_icf(bitmap_string, eu)

            for sample_id in test_ids:
                sample_dict = tests_sample[sample_id]["features"]
                sigmas = sigmas_all[sample_id]

                # Calculate cost
                cost = cost_function(
                    sample=sample_dict,
                    sigmas=sigmas,
                    icf=icf
                )

                results.append({
                    'sample_id': sample_id,
                    'bitmap_index': bitmap_string,
                    'cost': cost,
                    'icf': icf
                })
        except Exception as e:
            # Silent error - don't break parallel processing
            continue

    return batch_id, reason_type, results, processed_icfs


def calculate_costs_parallel_incremental(db, test_ids, tests_sample, sigmas_all, cost_function, bitmap_to_icf,
                                         reason_types, cache, selected_zip_path,
                                         n_workers=None, batch_size=50, save_every_n_batches=10, verbose=True,
                                         progress_monitor: Optional[ICFProgressMonitor] = None,
                                         cache_writer: Optional[CacheWriteCoordinator] = None):
    """
    Calculate costs in TRUE PARALLEL mode using ProcessPoolExecutor

    This uses REAL multiprocessing (separate processes), not threading.
    Each process runs on a separate CPU core, bypassing Python's GIL.

    Parameters
    ----------
    db : dict
        Database containing reasons data
    test_ids : list
        List of test sample IDs
    tests_sample : dict
        Dictionary of test samples
    sigmas_all : dict
        Sigmas for all test samples
    cost_function : callable
        Cost function to use
    bitmap_to_icf : callable
        Function to convert bitmap to ICF
    reason_types : list
        List of reason types to process
    cache : ETLCache
        Cache instance for incremental saving (with file locking)
    selected_zip_path : Path
        Path to the ZIP file for cache key
    n_workers : int, optional
        Number of parallel PROCESSES. Defaults to CPU count - 1
    batch_size : int
        Number of bitmaps to process per batch
    save_every_n_batches : int
        Save to cache every N batches to reduce RAM
    verbose : bool
        Print progress messages

    Returns
    -------
    tuple
        (total_costs_count, tests_sample_updated)
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)

    if verbose:
        print(f"\n🚀 TRUE MULTIPROCESSING MODE - Using {n_workers} SEPARATE PROCESSES")
        print(f"   Batch size: {batch_size} ICFs | Save every: {save_every_n_batches} batches")

    eu = db["data"]['EU']['value_json']
    total_costs = 0

    worker_init_args = (test_ids, tests_sample, sigmas_all,
                        cost_function, bitmap_to_icf, eu)

    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_cost_worker,
                             initargs=worker_init_args) as executor:

        for reason_idx, reason_type in enumerate(reason_types):
            if reason_type not in db or len(db[reason_type]) == 0:
                if verbose:
                    print(f"\n⚠️  {reason_type} not found in database or empty, skipping...")
                continue

            if verbose:
                print(f"\n{'='*80}")
                print(f"Processing {reason_type.upper()}")
                print(f"{'='*80}")
                print(f"  Total ICFs: {len(db[reason_type])}")

            bitmap_strings = list(db[reason_type].keys())

            # Split into batches
            batch_args = []
            for batch_idx in range(0, len(bitmap_strings), batch_size):
                batch = bitmap_strings[batch_idx:batch_idx+batch_size]
                batch_args.append((len(batch_args), reason_type, batch))

            # Pad with empty batches so the total is a multiple of n_workers
            if batch_args:
                remainder = len(batch_args) % n_workers
                if remainder != 0:
                    padding = n_workers - remainder
                    for _ in range(padding):
                        batch_args.append((len(batch_args), reason_type, []))
                    if verbose:
                        print(f"  Added {padding} padding batches to match {n_workers} workers")

            if verbose:
                print(f"  Created {len(batch_args)} batches")
                print(f"  Submitting ALL batches to {n_workers} parallel processes...")

            if progress_monitor:
                progress_monitor.start_reason(reason_type, len(batch_args), len(bitmap_strings))

            # Process ALL batches in TRUE PARALLEL using ProcessPoolExecutor
            accumulated_costs = []
            accumulated_samples = {}
            completed = 0
            icfs_this_type = 0  # Track ICFs for this reason type

            overall_start = time.time()

            # Submit ALL batches at once - they will run in parallel
            future_to_batch = {
                executor.submit(calculate_cost_for_icf_batch, batch_id, reason_type, batch): batch_id
                for batch_id, reason_type, batch in batch_args
            }

            # Process results as they complete
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]

                try:
                    returned_batch_id, returned_reason_type, batch_results, icfs_in_batch = future.result()
                    completed += 1
                    icfs_this_type += icfs_in_batch

                    # Accumulate results
                    for cost_entry in batch_results:
                        cost_entry['reason_type'] = returned_reason_type
                        accumulated_costs.append(cost_entry)

                        sample_id = cost_entry['sample_id']
                        bitmap_string = cost_entry['bitmap_index']

                        if sample_id not in accumulated_samples:
                            accumulated_samples[sample_id] = {}
                        if returned_reason_type not in accumulated_samples[sample_id]:
                            accumulated_samples[sample_id][returned_reason_type] = {}

                        accumulated_samples[sample_id][returned_reason_type][bitmap_string] = {
                            'icf': cost_entry['icf'],
                            'cost': cost_entry['cost']
                        }

                    if progress_monitor:
                        progress_monitor.batch_completed(returned_reason_type, icfs_in_batch)

                    if verbose:
                        elapsed = time.time() - overall_start
                        icf_rate = icfs_this_type / elapsed if elapsed > 0 else 0
                        print(f"  ✓ Batch {completed}/{len(batch_args)} complete | "
                              f"{icfs_this_type} ICFs | "
                              f"{icf_rate:.1f} ICF/sec")

                    # Save to cache incrementally
                    if completed % save_every_n_batches == 0 or completed == len(batch_args):
                        if cache and selected_zip_path and accumulated_costs:
                            is_first = (reason_idx == 0 and completed <= save_every_n_batches)

                            costs_to_save = accumulated_costs
                            samples_to_save = accumulated_samples
                            accumulated_costs = []
                            accumulated_samples = {}

                            if cache_writer:
                                cache_writer.submit(
                                    costs_to_save,
                                    samples_to_save,
                                    returned_reason_type,
                                    is_first_batch=is_first
                                )
                            else:
                                cache.save_costs_incremental(
                                    selected_zip_path,
                                    costs_to_save,
                                    samples_to_save,
                                    returned_reason_type,
                                    is_first_batch=is_first
                                )

                            if verbose:
                                saved_count = len(costs_to_save)
                                print(f"  💾 Saved {saved_count} costs to cache "
                                      f"(checkpoint at batch {completed}/{len(batch_args)})")

                            total_costs += saved_count

                            # Update main tests_sample
                            for sample_id, data in samples_to_save.items():
                                if sample_id not in tests_sample:
                                    tests_sample[sample_id] = {}
                                for rt, bitmaps in data.items():
                                    if rt not in tests_sample[sample_id]:
                                        tests_sample[sample_id][rt] = {}
                                    tests_sample[sample_id][rt].update(bitmaps)

                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Error in batch {batch_id}: {e}")

            total_elapsed = time.time() - overall_start
            if verbose:
                avg_icf_rate = icfs_this_type / total_elapsed if total_elapsed > 0 else 0
                print(f"\n  ✅ Completed {reason_type} in {total_elapsed:.1f}s")
                print(f"     Average: {avg_icf_rate:.1f} ICF/sec ({icfs_this_type} ICFs total)")
            if progress_monitor:
                progress_monitor.complete_reason(reason_type)

    if verbose:
        print(f"\n{'='*80}")
        print(f"🎉 ALL PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total costs calculated: {total_costs}")

    if cache_writer:
        cache_writer.flush()

    return total_costs, tests_sample

