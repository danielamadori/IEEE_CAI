"""
Raw data cache for database extractions.
Saves the raw db dictionaries (db0-db9) for each dataset.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional


class RawDataCache:
    """Cache for raw database data extracted from ZIP files"""

    def __init__(self, cache_dir: Path = None):
        """
        Initialize raw data cache

        Parameters
        ----------
        cache_dir : Path, optional
            Directory to store cache files. Defaults to results/_cache
        """
        if cache_dir is None:
            cache_dir = Path("results/_cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file
        self.meta_file = self.cache_dir / "raw_data_meta.json"
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

    def _get_cache_key(self, dataset_name: str) -> str:
        """Get cache key for a dataset"""
        # Remove .zip extension if present
        if dataset_name.endswith('.zip'):
            dataset_name = dataset_name[:-4]
        return dataset_name

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to raw data cache file"""
        return self.cache_dir / f"raw_{cache_key}.pkl"

    def exists(self, dataset_name: str) -> bool:
        """
        Check if raw data cache exists for a dataset

        Parameters
        ----------
        dataset_name : str
            Name of the dataset

        Returns
        -------
        bool
            True if cache exists
        """
        cache_key = self._get_cache_key(dataset_name)
        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()

    def save(self, dataset_name: str, db_data: Dict[str, Any]):
        """
        Save raw database data to cache

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        db_data : dict
            Dictionary containing all database data (db0-db9)
            Format: {
                'data': {...},      # db0
                'reasons': {...},   # db2
                'non_reasons': {...}, # db3
                'anti_reasons': {...}, # db5
                ...
            }
        """
        import shutil

        cache_key = self._get_cache_key(dataset_name)
        cache_path = self._get_cache_path(cache_key)

        # Check available disk space
        stat = shutil.disk_usage(self.cache_dir)
        available_mb = stat.free / (1024 * 1024)

        if available_mb < 100:  # Less than 100 MB available
            raise OSError(f"Not enough disk space to save cache (only {available_mb:.0f} MB available)")

        # Save data using pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(db_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update metadata
        self.metadata[cache_key] = {
            'dataset_name': dataset_name,
            'cached_at': str(cache_path.stat().st_mtime),
            'db_keys': list(db_data.keys()),
            'num_dbs': len(db_data)
        }
        self._save_metadata()

        print(f"Cached raw data for {dataset_name} ({len(db_data)} databases)")

    def load(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Load raw database data from cache

        Parameters
        ----------
        dataset_name : str
            Name of the dataset

        Returns
        -------
        dict or None
            Cached database data if exists, None otherwise
        """
        if not self.exists(dataset_name):
            return None

        cache_key = self._get_cache_key(dataset_name)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            num_dbs = len(data)
            db_names = ', '.join(list(data.keys())[:5])
            if num_dbs > 5:
                db_names += f', ... ({num_dbs} total)'

            print(f"📂 Loaded raw data from cache: {dataset_name}")
            print(f"   Databases: {db_names}")
            return data
        except (EOFError, pickle.UnpicklingError) as e:
            # Corrupted cache file - remove it
            print(f"✗ Corrupted raw cache detected for {dataset_name}: {e}")
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
            print(f"✗ Error loading raw cache for {dataset_name}: {e}")
            return None

    def clear(self, dataset_name: Optional[str] = None):
        """
        Clear raw data cache

        Parameters
        ----------
        dataset_name : str, optional
            If provided, clear cache only for this dataset.
            If None, clear all raw data cache.
        """
        if dataset_name is None:
            # Clear all raw data cache
            for cache_file in self.cache_dir.glob("raw_*.pkl"):
                cache_file.unlink()
            self.metadata = {}
            self._save_metadata()
            print("✓ Cleared all raw data cache")
        else:
            # Clear specific cache
            cache_key = self._get_cache_key(dataset_name)
            cache_path = self._get_cache_path(cache_key)

            if cache_path.exists():
                cache_path.unlink()

            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()

            print(f"✓ Cleared raw data cache for {dataset_name}")

    def list_cached(self) -> Dict[str, Dict[str, Any]]:
        """
        List all cached datasets

        Returns
        -------
        dict
            Dictionary mapping cache keys to their metadata
        """
        return self.metadata.copy()

    def get_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about cached dataset without loading it

        Parameters
        ----------
        dataset_name : str
            Name of the dataset

        Returns
        -------
        dict or None
            Metadata if cached, None otherwise
        """
        cache_key = self._get_cache_key(dataset_name)
        return self.metadata.get(cache_key)

