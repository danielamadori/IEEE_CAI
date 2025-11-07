"""
Cache system for ETL results to speed up notebook execution.
Saves extracted database information to avoid re-processing ZIP files every time.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib


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
    """Cache manager for ETL results"""

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
        """Get path to cache file"""
        return self.cache_dir / f"{cache_key}.pkl"

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
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
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
            - workers_table: worker statistics table
            - workers_table2: worker statistics table 2
            - workers_table3: worker statistics table 3
            - plots: plot data
        """
        cache_key = self._get_cache_key(zip_path.name)
        cache_path = self._get_cache_path(cache_key)
        file_hash = self._get_file_hash(zip_path)

        # Sanifica i dati per evitare errori di pickle con moduli/figure/funzioni
        safe_data = _make_pickleable(data)

        # Save data using pickle for speed
        with open(cache_path, 'wb') as f:
            pickle.dump(safe_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update metadata
        self.metadata[cache_key] = {
            'file_hash': file_hash,
            'file_name': zip_path.name,
            'cached_at': str(Path(cache_path).stat().st_mtime)
        }
        self._save_metadata()

        print(f"✓ Cached data for {zip_path.name}")

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
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded cached data for {zip_path.name}")
            return data
        except Exception as e:
            print(f"✗ Error loading cache for {zip_path.name}: {e}")
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
        if zip_name is None:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.metadata = {}
            self._save_metadata()
            print("✓ Cleared all cache")
        else:
            # Clear specific cache
            cache_key = self._get_cache_key(zip_name)
            cache_path = self._get_cache_path(cache_key)

            if cache_path.exists():
                cache_path.unlink()

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

