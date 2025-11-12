"""
Cache system for ETL results to speed up notebook execution.
Saves extracted database information to avoid re-processing ZIP files every time.
Thread-safe with file locking to prevent race conditions in parallel processing.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import time
import gzip


def _make_pickleable(obj: Any) -> Any:
    """
    Trasforma ricorsivamente l'oggetto in qualcosa di pickleable:
    - dizionari, liste, tuple, set vengono processati ricorsivamente
    - oggetti non serializzabili vengono sostituiti con None
    """
    try:
        # rapido controllo: se è già pickleable, restituiscilo così com'è
        pickle.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {k: _make_pickleable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [_make_pickleable(v) for v in obj]
            return type(obj)(converted)
        if isinstance(obj, set):
            return set(_make_pickleable(v) for v in obj)
        # fallback: oggetti complessi (es. moduli, figure, funzioni) -> None
        return None


class ETLCache:
    """Cache manager for ETL results with thread-safe locking"""

    def __init__(self, cache_dir: Path = None):
        """
        Initialize cache manager

        Parameters
        ----------
        cache_dir : Path, optional
            Directory to store cache files. Defaults to results/_cache
        """
        if cache_dir is None:
            cache_dir = Path("results/_cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file to track cache versions
        self.meta_file = self.cache_dir / "cache_meta.json"
        self._load_metadata()

        # Lock directory for file locks
        self.lock_dir = self.cache_dir / "locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self):
        """Load cache metadata"""
        if self.meta_file.exists():
            with open(self.meta_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.meta_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file for cache validation"""
        hasher = hashlib.md5()

        # Hash file size and modification time for speed
        stat = file_path.stat()
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime).encode())

        return hasher.hexdigest()

    def _get_cache_key(self, zip_name: str) -> str:
        """Get cache key for a dataset"""
        # Remove .zip extension if present
        if zip_name.endswith('.zip'):
            zip_name = zip_name[:-4]
        return zip_name

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to compressed cache file"""
        return self.cache_dir / f"{cache_key}.pkl.gz"

    def _get_legacy_cache_path(self, cache_key: str) -> Path:
        """Get path to legacy (uncompressed) cache file"""
        return self.cache_dir / f"{cache_key}.pkl"

    def _locate_cache_file(self, cache_key: str) -> Optional[Path]:
        """Return existing cache file path, preferring compressed version"""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            return cache_path

        legacy_path = self._get_legacy_cache_path(cache_key)
        if legacy_path.exists():
            return legacy_path
        return None

    def _open_for_read(self, cache_path: Path):
        """Open cache file for reading with or without compression"""
        if cache_path.suffix == '.gz':
            return gzip.open(cache_path, 'rb')
        return open(cache_path, 'rb')

    def _open_for_write(self, cache_path: Path):
        """Open cache file for writing (compressed)"""
        if cache_path.suffix == '.gz':
            return gzip.open(cache_path, 'wb', compresslevel=5)
        return open(cache_path, 'wb')

    def _get_lock_path(self, cache_key: str) -> Path:
        """Get path to lock file for a cache key"""
        return self.lock_dir / f"{cache_key}.lock"

    def _acquire_lock(self, cache_key: str, timeout: float = 30.0):
        """
        Acquire exclusive lock for cache file

        Parameters
        ----------
        cache_key : str
            Cache key to lock
        timeout : float
            Maximum seconds to wait for lock

        Returns
        -------
        file handle or None
            Lock file handle if acquired, None if timeout
        """
        import sys
        lock_path = self._get_lock_path(cache_key)
        lock_file = open(lock_path, 'w')

        start_time = time.time()
        while True:
            try:
                if sys.platform == 'win32':
                    # Windows: use msvcrt
                    import msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Unix: use fcntl
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_file
            except (IOError, OSError):
                if time.time() - start_time > timeout:
                    lock_file.close()
                    return None
                time.sleep(0.1)  # Wait 100ms before retry

    def _release_lock(self, lock_file):
        """Release lock and close file"""
        if lock_file:
            try:
                import sys
                if sys.platform == 'win32':
                    import msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except:
                pass
            finally:
                lock_file.close()

    def is_valid(self, zip_path: Path) -> bool:
        """
        Check if cache is valid for a given ZIP file

        Parameters
        ----------
        zip_path : Path
            Path to the ZIP file

        Returns
        -------
        bool
            True if cache is valid, False otherwise
        """
        cache_key = self._get_cache_key(zip_path.name)
        cache_path = self._locate_cache_file(cache_key)

        if cache_path is None:
            return False

        # Check if hash matches
        current_hash = self._get_file_hash(zip_path)
        cached_hash = self.metadata.get(cache_key, {}).get('file_hash')

        return current_hash == cached_hash

    def save(self, zip_path: Path, data: Dict[str, Any]):
        """
        Save ETL results to cache

        Parameters
        ----------
        zip_path : Path
            Path to the ZIP file
        data : dict
            Dictionary containing:
            - db: all database contents (db0-db9)
            - costs: (optional) calculated costs DataFrame and tests_sample data
            - workers_table: worker statistics table
            - workers_table2: worker statistics table 2
            - workers_table3: worker statistics table 3
            - plots: plot data
        """
        import shutil

        cache_key = self._get_cache_key(zip_path.name)
        cache_path = self._get_cache_path(cache_key)
        legacy_path = self._get_legacy_cache_path(cache_key)
        file_hash = self._get_file_hash(zip_path)

        # Check available disk space
        stat = shutil.disk_usage(self.cache_dir)
        available_mb = stat.free / (1024 * 1024)

        if available_mb < 100:  # Less than 100 MB available
            raise OSError(f"Not enough disk space to save cache (only {available_mb:.0f} MB available)")

        # Sanifica i dati per evitare errori di pickle con moduli/figure/funzioni
        safe_data = _make_pickleable(data)

        # Acquire lock before saving
        lock_file = self._acquire_lock(cache_key)
        if not lock_file:
            raise RuntimeError(f"Failed to acquire lock for {zip_path.name} after waiting")

        try:
            # Save data using compressed pickle for better disk usage
            with self._open_for_write(cache_path) as f:
                pickle.dump(safe_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Remove legacy uncompressed cache if it exists to keep one file per dataset
            if legacy_path.exists() and legacy_path != cache_path:
                try:
                    legacy_path.unlink()
                except OSError:
                    pass

            # Update metadata
            self.metadata[cache_key] = {
                'file_hash': file_hash,
                'file_name': zip_path.name,
                'cached_at': str(Path(cache_path).stat().st_mtime)
            }
            self._save_metadata()

            print(f"✓ Cached data for {zip_path.name}")
        except Exception as e:
            print(f"Error saving cache for {zip_path.name}: {e}")
        finally:
            # Release lock
            self._release_lock(lock_file)

    def load(self, zip_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load ETL results from cache

        Parameters
        ----------
        zip_path : Path
            Path to the ZIP file

        Returns
        -------
        dict or None
            Cached data if valid, None otherwise
        """
        if not self.is_valid(zip_path):
            return None

        cache_key = self._get_cache_key(zip_path.name)
        cache_path = self._locate_cache_file(cache_key)

        if cache_path is None:
            return None

        try:
            with self._open_for_read(cache_path) as f:
                data = pickle.load(f)
            print(f"[OK] Loaded cached data for {zip_path.name}")
            return data
        except (EOFError, pickle.UnpicklingError) as e:
            # Corrupted cache file - remove it
            print(f"[ERR] Corrupted cache detected for {zip_path.name}: {e}")
            print(f"   Removing corrupted cache file...")
            try:
                cache_path.unlink()
                # Also remove from metadata
                if cache_key in self.metadata:
                    del self.metadata[cache_key]
                    self._save_metadata()
            except Exception:
                pass
            return None
        except Exception as e:
            print(f"[ERR] Error loading cache for {zip_path.name}: {e}")
            return None

    def clear(self, zip_name: Optional[str] = None):
        """
        Clear cache

        Parameters
        ----------
        zip_name : str, optional
            If provided, clear cache only for this dataset.
            If None, clear all cache.
        """
        def _delete_files(pattern: str):
            for cache_file in self.cache_dir.glob(pattern):
                try:
                    cache_file.unlink()
                except FileNotFoundError:
                    pass

        if zip_name is None:
            # Clear all cache
            _delete_files("*.pkl")
            _delete_files("*.pkl.gz")
            self.metadata = {}
            self._save_metadata()
            print("✓ Cleared all cache")
        else:
            # Clear specific cache
            cache_key = self._get_cache_key(zip_name)
            for path in (self._get_cache_path(cache_key), self._get_legacy_cache_path(cache_key)):
                if path.exists():
                    path.unlink()

            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()

            print(f"✓ Cleared cache for {zip_name}")

    def list_cached(self) -> Dict[str, Dict[str, Any]]:
        """
        List all cached datasets

        Returns
        -------
        dict
            Dictionary mapping cache keys to their metadata
        """
        return self.metadata.copy()

    def save_costs(self, zip_path: Path, cost_df, tests_sample: Dict[str, Any], reason_types: list):
        """
        Save costs calculation results to existing cache

        Parameters
        ----------
        zip_path : Path
            Path to the ZIP file
        cost_df : pd.DataFrame
            DataFrame with calculated costs
        tests_sample : dict
            Dictionary with test samples and their costs
        reason_types : list
            List of reason types that were calculated
        """
        cache_key = self._get_cache_key(zip_path.name)
        existing_path = self._locate_cache_file(cache_key)
        target_path = self._get_cache_path(cache_key)

        # Load existing cache
        if existing_path is None:
            print(f"Warning: No cache exists for {zip_path.name}, cannot save costs")
            return

        try:
            with self._open_for_read(existing_path) as f:
                cached_data = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing cache for {zip_path.name}: {e}")
            return

        # Add costs to cached data
        cached_data['costs'] = {
            'cost_df': cost_df,
            'tests_sample': tests_sample,
            'reason_types': reason_types
        }

        # Make pickleable and save
        safe_data = _make_pickleable(cached_data)

        # Acquire lock before saving costs
        lock_file = self._acquire_lock(cache_key)
        if not lock_file:
            raise RuntimeError(f"Failed to acquire lock for {zip_path.name} after waiting")

        try:
            with self._open_for_write(target_path) as f:
                pickle.dump(safe_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            legacy_path = self._get_legacy_cache_path(cache_key)
            if legacy_path.exists() and legacy_path != target_path:
                try:
                    legacy_path.unlink()
                except OSError:
                    pass
            print(f"✓ Saved costs to cache for {zip_path.name} ({len(cost_df)} entries)")
        except Exception as e:
            print(f"Warning: Could not save costs to cache: {e}")
        finally:
            # Release lock
            self._release_lock(lock_file)

    def load_costs(self, zip_path: Path, reason_types: list = None):
        """
        Load costs from cache if available

        Parameters
        ----------
        zip_path : Path
            Path to the ZIP file
        reason_types : list, optional
            List of reason types to check. If provided, validates that all types are present.

        Returns
        -------
        dict or None
            Dictionary with 'cost_df' and 'tests_sample' if available, None otherwise
        """
        cache_key = self._get_cache_key(zip_path.name)
        cache_path = self._locate_cache_file(cache_key)

        if cache_path is None:
            return None

        try:
            with self._open_for_read(cache_path) as f:
                cached_data = pickle.load(f)

            # Check if costs are present
            if 'costs' not in cached_data:
                return None

            costs_data = cached_data['costs']

            # Validate reason types if provided
            if reason_types is not None:
                cached_types = set(costs_data.get('reason_types', []))
                requested_types = set(reason_types)
                if not requested_types.issubset(cached_types):
                    return None

            return {
                'cost_df': costs_data['cost_df'],
                'tests_sample': costs_data['tests_sample']
            }
        except Exception as e:
            print(f"Warning: Could not load costs from cache: {e}")
            return None

    def has_costs(self, zip_path: Path) -> bool:
        """
        Check if cache has costs data

        Parameters
        ----------
        zip_path : Path
            Path to the ZIP file

        Returns
        -------
        bool
            True if costs are cached, False otherwise
        """
        cache_key = self._get_cache_key(zip_path.name)
        cache_path = self._locate_cache_file(cache_key)

        if cache_path is None:
            return False

        try:
            with self._open_for_read(cache_path) as f:
                cached_data = pickle.load(f)
            return 'costs' in cached_data and cached_data['costs'] is not None
        except Exception:
            return False

    def save_costs_incremental(self, zip_path: Path, cost_batch, tests_sample_batch: Dict[str, Any],
                               reason_type: str, is_first_batch: bool = False):
        """
        Save costs incrementally to avoid excessive RAM usage (THREAD-SAFE)

        Parameters
        ----------
        zip_path : Path
            Path to the ZIP file
        cost_batch : list or pd.DataFrame
            Batch of cost data to append
        tests_sample_batch : dict
            Batch of test samples with their costs
        reason_type : str
            Type of reason being processed (reasons, non_reasons, anti_reasons)
        is_first_batch : bool
            If True, initialize the costs structure in cache
        """
        import pandas as pd

        cache_key = self._get_cache_key(zip_path.name)
        target_path = self._get_cache_path(cache_key)
        existing_path = self._locate_cache_file(cache_key)

        # Load existing cache
        if existing_path is None:
            print(f"Warning: No cache exists for {zip_path.name}, cannot save costs incrementally")
            return

        # Acquire lock before read-modify-write
        lock_file = self._acquire_lock(cache_key, timeout=60.0)  # Longer timeout for parallel workers
        if not lock_file:
            print(f"Warning: Could not acquire lock for {zip_path.name}, skipping incremental save")
            return

        try:
            # Load existing cache inside lock
            with self._open_for_read(existing_path) as f:
                cached_data = pickle.load(f)

            # Initialize costs structure if first batch
            if is_first_batch and 'costs' not in cached_data:
                cached_data['costs'] = {
                    'cost_df': pd.DataFrame(),
                    'tests_sample': {},
                    'reason_types': []
                }

            # Append cost data
            if 'costs' in cached_data:
                # Convert batch to DataFrame if needed
                if isinstance(cost_batch, list):
                    cost_batch_df = pd.DataFrame(cost_batch)
                else:
                    cost_batch_df = cost_batch

                # Append to existing DataFrame
                if cached_data['costs']['cost_df'].empty:
                    cached_data['costs']['cost_df'] = cost_batch_df
                else:
                    cached_data['costs']['cost_df'] = pd.concat(
                        [cached_data['costs']['cost_df'], cost_batch_df],
                        ignore_index=True
                    )

                # Update tests_sample
                for sample_id, data in tests_sample_batch.items():
                    if sample_id not in cached_data['costs']['tests_sample']:
                        cached_data['costs']['tests_sample'][sample_id] = {}
                    cached_data['costs']['tests_sample'][sample_id].update(data)

                # Add reason type if not present
                if reason_type not in cached_data['costs']['reason_types']:
                    cached_data['costs']['reason_types'].append(reason_type)

            # Make pickleable and save
            safe_data = _make_pickleable(cached_data)

            with self._open_for_write(target_path) as f:
                pickle.dump(safe_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            legacy_path = self._get_legacy_cache_path(cache_key)
            if legacy_path.exists() and legacy_path != target_path:
                try:
                    legacy_path.unlink()
                except OSError:
                    pass

        except Exception as e:
            print(f"Warning: Could not save incremental costs: {e}")
        finally:
            # Release lock
            self._release_lock(lock_file)
