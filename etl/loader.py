import json
from pathlib import Path
import zipfile, base64

from etl.data_loader import load_db0, render_db0_sample_timeseries
from etl.logs_loader import build_db10_worker_report
from etl.cache import ETLCache
from etl.raw_cache import RawDataCache
from etl.zip_inspector import scan_and_load, collect_archive_data, decode_key, try_decode_value
from etl.constants import DB_NAMES

# Global cache instances
_cache = ETLCache()
_raw_cache = RawDataCache()


def etl(zip_paths, results_dir, use_cache=True, force_refresh=False, auto_select=False, skip_workers_report=False, load_only_db10=False, verbose=True):
    """
    Extract, Transform, Load data from ZIP archives with 2-level caching

    Parameters
    ----------
    zip_paths : list
        List of paths to ZIP files
    results_dir : Path
        Directory containing results
    use_cache : bool, optional
        Whether to use cache (default: True)
    force_refresh : bool, optional
        Force refresh Full Cache but use Raw Cache if available (default: False)
    auto_select : bool, optional
        If True, automatically select the first dataset without prompting (default: False)
    skip_workers_report : bool, optional
        If True, skip building the workers report (DB10) to save time (default: False)
    load_only_db10 : bool, optional
        If True, load only DB10 (LOGS) without DB0-DB9 to save RAM (default: False)
        Note: skip_workers_report is ignored when load_only_db10=True
    verbose : bool, optional
        If True, print progress messages (default: True)

    Returns
    -------
    dict
        Database dictionary with all extracted data
    """
    selected_zip_name = scan_and_load(zip_paths, results_dir, auto_select=auto_select, verbose=verbose)

    # Find the selected ZIP path
    selected_zip_path = None
    for path in zip_paths:
        if path.name == selected_zip_name:
            selected_zip_path = path
            break

    # Special mode: load only DB10 (LOGS) to save RAM
    if load_only_db10:
        if verbose:
            print(f"Loading only DB10 (LOGS) for {selected_zip_name}...")
        db = {}
        archives_data = [collect_archive_data(path) for path in zip_paths]
        manifests_by_archive = {item['zip_name']: item['manifest'] for item in archives_data}
        manifest_prefix_by_archive = {item['zip_name']: item.get('manifest_prefix', '') for item in archives_data}
        backups_by_archive = {item['zip_name']: item['backups'] for item in archives_data}
        selected_archive_data = next((item for item in archives_data if item['zip_name'] == selected_zip_name), None)
        selected_manifest = manifests_by_archive.get(selected_zip_name)
        selected_manifest_prefix = manifest_prefix_by_archive.get(selected_zip_name, '')
        selected_backups = backups_by_archive.get(selected_zip_name)

        db[DB_NAMES.get(10)] = build_db10_worker_report(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix, max_events=None)

        # Extract dataset name from ZIP filename (e.g., "Coffee_0_false_0.zip" -> "Coffee")
        dataset_name = selected_zip_name.split('_')[0] if '_' in selected_zip_name else selected_zip_name.replace('.zip', '')
        db['_dataset_name'] = dataset_name

        return db

    # Level 2: Try Full Cache first (if not force_refresh)
    # Note: Full Cache only stores 'db', workers and plots are regenerated
    if use_cache and not force_refresh and selected_zip_path:
        cached_data = _cache.load(selected_zip_path)
        if cached_data is not None:
            if verbose:
                print(f"Using cached DB" + (", skipping workers report..." if skip_workers_report else ", regenerating workers/plots..."))
            db = cached_data['db']

            # Regenerate workers and plots (not cacheable due to unpicklable objects)
            if not skip_workers_report:
                # Process only the selected ZIP, not all ZIPs
                selected_archive_data = collect_archive_data(selected_zip_path)
                selected_manifest = selected_archive_data['manifest']
                selected_manifest_prefix = selected_archive_data.get('manifest_prefix', '')
                selected_backups = selected_archive_data['backups']

                db[DB_NAMES.get(10)] = build_db10_worker_report(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix, max_events=None)

            # Extract dataset name from ZIP filename
            dataset_name = selected_zip_name.split('_')[0] if '_' in selected_zip_name else selected_zip_name.replace('.zip', '')
            db['_dataset_name'] = dataset_name

            return db

    if use_cache and verbose:
        print(f"Processing {selected_zip_name}...")

    # Level 1: Try Raw Cache for database extraction
    db = None
    if use_cache and not force_refresh:
        db = _raw_cache.load(selected_zip_name)

    # If no raw cache, extract databases from ZIP
    if db is None:
        db = _extract_databases(zip_paths, selected_zip_name, results_dir)

        # Save to Raw Cache
        if use_cache:
            try:
                _raw_cache.save(selected_zip_name, db)
            except OSError as e:
                if verbose:
                    print(f"Warning: Could not save raw cache: {e}")
                    print("   Continuing without cache...")
            except Exception as e:
                if verbose:
                    print(f"Warning: Unexpected error saving raw cache: {e}")

    # Process workers and plots (not cached at raw level)
    if not skip_workers_report:
        # Process only the selected ZIP, not all ZIPs
        selected_archive_data = collect_archive_data(selected_zip_path)
        selected_manifest = selected_archive_data['manifest']
        selected_manifest_prefix = selected_archive_data.get('manifest_prefix', '')
        selected_backups = selected_archive_data['backups']

        db[DB_NAMES.get(10)] = build_db10_worker_report(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix, max_events=None)

    render_db0_sample_timeseries(db.get(DB_NAMES.get(0), {}))

    # Extract dataset name from ZIP filename
    dataset_name = selected_zip_name.split('_')[0] if '_' in selected_zip_name else selected_zip_name.replace('.zip', '')
    db['_dataset_name'] = dataset_name

    # Save to Full Cache (only db, workers/plots are regenerated on load)
    if use_cache and selected_zip_path:
        cache_data = {
            'db': db,
        }
        try:
            _cache.save(selected_zip_path, cache_data)
        except OSError as e:
            if verbose:
                print(f"Warning: Could not save full cache: {e}")
                print("   Continuing without cache...")
        except Exception as e:
            if verbose:
                print(f"Warning: Unexpected error saving cache: {e}")

    return db


def _extract_databases(zip_paths, selected_zip_name, results_dir):
    """
    Extract all databases from the selected ZIP file

    Parameters
    ----------
    zip_paths : list
        List of all ZIP paths
    selected_zip_name : str
        Name of the selected ZIP file
    results_dir : Path
        Results directory

    Returns
    -------
    dict
        Dictionary with all database data
    """
    print(f"Extracting databases from {selected_zip_name}...")

    # Find the selected ZIP path
    selected_zip_path = None
    for path in zip_paths:
        if path.name == selected_zip_name:
            selected_zip_path = path
            break

    if selected_zip_path is None:
        raise ValueError(f"ZIP file not found: {selected_zip_name}")

    # Process only the selected ZIP, not all ZIPs
    selected_archive_data = collect_archive_data(selected_zip_path)
    selected_manifest = selected_archive_data['manifest']
    selected_manifest_prefix = selected_archive_data.get('manifest_prefix', '')
    selected_backups = selected_archive_data['backups']

    db = {DB_NAMES.get(0): load_db0(selected_manifest, selected_backups)}

    if selected_manifest and selected_archive_data and selected_manifest_prefix is not None:
        files_map = (selected_manifest.get('files') or {})
        selected_zip_path = Path(selected_archive_data['zip_path'])
        with zipfile.ZipFile(selected_zip_path) as z:
            for db_index in range(1, 10):
                file_name = files_map.get(str(db_index))
                if not file_name:
                    continue
                member_name = f"{selected_manifest_prefix}{file_name}"
                try:
                    raw_text = z.read(member_name).decode("utf-8", errors="replace")
                    obj = json.loads(raw_text)
                except Exception:
                    db[db_index] = {'file_name': file_name, 'entries': []}
                    continue

                entries_map = {}
                for entry in obj.get('entries') or []:
                    try:
                        key_bytes = decode_key(entry)
                        key_text = key_bytes.decode('utf-8', errors='replace')
                    except Exception as exc:
                        key_text = f'<unable to decode key: {exc}>'

                    preview, details = try_decode_value(entry)
                    value_bytes = details.get('decoded_bytes') if isinstance(details, dict) else None

                    if not isinstance(value_bytes, (bytes, bytearray)):
                        v = entry.get('value') or {}
                        if isinstance(v, dict) and isinstance(v.get('data'), str):
                            try:
                                value_bytes = base64.b64decode(v['data'], validate=False)
                            except Exception:
                                value_bytes = None

                    if isinstance(value_bytes, (bytes, bytearray)):
                        plain_value = value_bytes.decode('utf-8', errors='replace')
                    else:
                        plain_value = str(preview)

                    try:
                        entries_map[key_text] = json.loads(plain_value)
                    except Exception:
                        entries_map[key_text] = plain_value

                name = DB_NAMES.get(db_index, f"db_{db_index}")
                db[name] = entries_map

    return db





def clear_raw_cache(dataset_name=None):
    """Clear raw database cache"""
    _raw_cache.clear(dataset_name)


def list_raw_cache():
    """List cached raw databases"""
    return _raw_cache.list_cached()


def clear_cache(dataset_name=None):
    """Clear full ETL cache"""
    _cache.clear(dataset_name)


def list_cache():
    """List cached datasets"""
    return _cache.list_cached()

