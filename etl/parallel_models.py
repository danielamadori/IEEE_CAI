"""
Parallel model analysis for models_analysis.ipynb
Extracts accuracy and parameters from multiple ZIP files in parallel.
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import multiprocessing


def extract_model_accuracy_worker(zip_path: Path) -> Dict[str, Any]:
    """
    Worker function to extract accuracy from a single ZIP file

    Parameters
    ----------
    zip_path : Path
        Path to ZIP file

    Returns
    -------
    dict
        {
            'dataset_name': str,
            'zip_name': str,
            'test_accuracy': float or None,
            'cv_score': float or None,
            'best_params': dict or None,
            'error': str or None
        }
    """
    try:
        from etl.zip_inspector import collect_archive_data
        from etl.data_loader import load_db0

        # Extract dataset name
        dataset_name = zip_path.stem.split('_')[0] if '_' in zip_path.stem else zip_path.stem

        # Load only DB0 (lightweight)
        archive_data = collect_archive_data(zip_path)
        manifest = archive_data.get('manifest')
        backups = archive_data.get('backups')

        result = {
            'dataset_name': dataset_name,
            'zip_name': zip_path.name,
            'test_accuracy': None,
            'cv_score': None,
            'best_params': None,
            'error': None
        }

        if not manifest or not backups:
            result['error'] = 'No manifest or backups'
            return result

        # Load DB0
        db0_data = load_db0(manifest, backups)

        # Extract RF_OPTIMIZATION_RESULTS
        rf_opt_entry = db0_data.get('RF_OPTIMIZATION_RESULTS', {})
        rf_opt_data = rf_opt_entry.get('value_json', {})

        if rf_opt_data:
            try:
                result['test_accuracy'] = float(rf_opt_data.get('test_score')) if rf_opt_data.get('test_score') is not None else None
                result['cv_score'] = float(rf_opt_data.get('best_cv_score')) if rf_opt_data.get('best_cv_score') is not None else None
                result['best_params'] = rf_opt_data.get('best_params')
            except (TypeError, ValueError) as e:
                result['error'] = f'Parse error: {e}'
        else:
            result['error'] = 'No RF_OPTIMIZATION_RESULTS found'

        return result

    except Exception as e:
        return {
            'dataset_name': zip_path.stem.split('_')[0] if '_' in zip_path.stem else zip_path.stem,
            'zip_name': zip_path.name,
            'test_accuracy': None,
            'cv_score': None,
            'best_params': None,
            'error': str(e)
        }


def extract_all_models_accuracy_parallel(zip_paths: list, n_workers=None, verbose=True) -> Dict[str, Dict[str, Any]]:
    """
    Extract accuracy from all ZIP files in parallel

    Parameters
    ----------
    zip_paths : list
        List of Path objects to ZIP files
    n_workers : int, optional
        Number of parallel workers (default: CPU count - 1)
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Dictionary mapping dataset_name -> accuracy_data
    """
    from multiprocessing import Pool, cpu_count

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    if verbose:
        print(f"\n{'='*80}")
        print(f"PARALLEL ACCURACY EXTRACTION")
        print(f"{'='*80}")
        print(f"Processing {len(zip_paths)} ZIP files with {n_workers} workers...")

    # Process in parallel
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(extract_model_accuracy_worker, zip_paths)
    else:
        results = [extract_model_accuracy_worker(zp) for zp in zip_paths]

    # Build dictionary
    accuracy_map = {}
    success_count = 0
    error_count = 0
    error_datasets = []

    for result in results:
        dataset_name = result['dataset_name']
        accuracy_map[dataset_name] = result
        
        if result['error']:
            error_count += 1
            error_datasets.append(dataset_name)
        elif result['test_accuracy'] is not None:
            success_count += 1

    if verbose:
        print(f"\n✓ Extraction complete:")
        print(f"  - Success: {success_count}/{len(zip_paths)}")
        print(f"  - Errors: {error_count}/{len(zip_paths)}, datasets errors {error_datasets}")
        print(f"{'='*80}\n")

    return accuracy_map


def extract_accuracy_for_analyzed_datasets(analyzed_sources: set, results_dir: Path,
                                           n_workers=None, verbose=True) -> Dict[str, Dict]:
    """
    Extract accuracy only for analyzed datasets (those with ZIP files)

    Parameters
    ----------
    analyzed_sources : set
        Set of dataset names that have been analyzed
    results_dir : Path
        Directory containing ZIP files
    n_workers : int, optional
        Number of parallel workers
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Dictionary mapping dataset_name -> {test_accuracy, cv_score, best_params}
    """
    from pathlib import Path

    # Find ZIP files for analyzed datasets
    zip_paths = []
    for dataset_name in analyzed_sources:
        pattern = f"{dataset_name}_*.zip"
        matches = list(results_dir.glob(pattern))
        if matches:
            zip_paths.append(matches[0])  # Take first match

    if not zip_paths:
        if verbose:
            print("No ZIP files found for analyzed datasets")
        return {}

    # Extract in parallel
    accuracy_map = extract_all_models_accuracy_parallel(zip_paths, n_workers=n_workers, verbose=verbose)

    return accuracy_map

