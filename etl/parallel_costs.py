"""
Parallel cost calculation for reasons analysis - TRUE MULTIPROCESSING
Uses ProcessPoolExecutor for real parallel execution across multiple CPU cores.
Each batch runs in a separate process (no GIL limitation).
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Tuple
import time


def calculate_cost_for_icf_batch(args: Tuple) -> Tuple[int, str, List[Dict]]:
    """
    Calculate costs for a batch of ICFs across all samples

    This function runs in a SEPARATE PROCESS (true parallelism, no GIL)

    Parameters
    ----------
    args : tuple
        (batch_id, reason_type, bitmap_strings, eu, test_ids, tests_sample, sigmas_all, cost_function, bitmap_to_icf)

    Returns
    -------
    tuple
        (batch_id, reason_type, results_list)
    """
    batch_id, reason_type, bitmap_strings, eu, test_ids, tests_sample, sigmas_all, cost_function, bitmap_to_icf = args

    results = []
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

    return batch_id, reason_type, results


def calculate_costs_parallel_incremental(db, test_ids, tests_sample, sigmas_all, cost_function, bitmap_to_icf,
                                         reason_types, cache, selected_zip_path,
                                         n_workers=None, batch_size=50, save_every_n_batches=10, verbose=True):
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
        n_workers = max(1, cpu_count() - 1)

    if verbose:
        print(f"\n🚀 TRUE MULTIPROCESSING MODE - Using {n_workers} SEPARATE PROCESSES")
        print(f"   Each process runs on a separate CPU core (no GIL limitation)")
        print(f"   Batch size: {batch_size} ICFs | Save every: {save_every_n_batches} batches")

    eu = db["data"]['EU']['value_json']
    total_costs = 0

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
            batch_args.append((
                len(batch_args),  # batch_id
                reason_type,
                batch,
                eu,
                test_ids,
                tests_sample,
                sigmas_all,
                cost_function,
                bitmap_to_icf
            ))

        if verbose:
            print(f"  Created {len(batch_args)} batches")
            print(f"  Submitting ALL batches to {n_workers} parallel processes...")

        # Process ALL batches in TRUE PARALLEL using ProcessPoolExecutor
        accumulated_costs = []
        accumulated_samples = {}
        completed = 0

        overall_start = time.time()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit ALL batches at once - they will run in parallel
            future_to_batch = {executor.submit(calculate_cost_for_icf_batch, args): args[0]
                              for args in batch_args}

            # Process results as they complete
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]

                try:
                    returned_batch_id, returned_reason_type, batch_results = future.result()
                    completed += 1

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

                    if verbose:
                        elapsed = time.time() - overall_start
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(f"  ✓ Batch {completed}/{len(batch_args)} complete | "
                              f"{len(batch_results)} costs | "
                              f"{rate:.1f} batches/sec")

                    # Save to cache incrementally
                    if completed % save_every_n_batches == 0 or completed == len(batch_args):
                        if cache and selected_zip_path and accumulated_costs:
                            is_first = (reason_idx == 0 and completed <= save_every_n_batches)

                            cache.save_costs_incremental(
                                selected_zip_path,
                                accumulated_costs,
                                accumulated_samples,
                                returned_reason_type,
                                is_first_batch=is_first
                            )

                            if verbose:
                                print(f"  💾 Saved {len(accumulated_costs)} costs to cache "
                                      f"(checkpoint at batch {completed}/{len(batch_args)})")

                            total_costs += len(accumulated_costs)

                            # Update main tests_sample
                            for sample_id, data in accumulated_samples.items():
                                if sample_id not in tests_sample:
                                    tests_sample[sample_id] = {}
                                for rt, bitmaps in data.items():
                                    if rt not in tests_sample[sample_id]:
                                        tests_sample[sample_id][rt] = {}
                                    tests_sample[sample_id][rt].update(bitmaps)

                            # Clear accumulated data to free RAM
                            accumulated_costs = []
                            accumulated_samples = {}

                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Error in batch {batch_id}: {e}")

        total_elapsed = time.time() - overall_start
        if verbose:
            print(f"\n  ✅ Completed {reason_type} in {total_elapsed:.1f}s")
            print(f"     Average: {len(batch_args)/total_elapsed:.2f} batches/sec")

    if verbose:
        print(f"\n{'='*80}")
        print(f"🎉 ALL PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total costs calculated: {total_costs}")

    return total_costs, tests_sample

