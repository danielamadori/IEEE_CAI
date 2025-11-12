"""
ETL Cache Manager - Comprehensive cache management tool

Manage, monitor, build and clear ETL cache for datasets.

Usage:
    python cache_manager.py --info                           # Show cache statistics
    python cache_manager.py --list                           # List all cached datasets
    python cache_manager.py --build                          # Build cache for all datasets
    python cache_manager.py --build --dataset NAME           # Build cache for specific dataset
    python cache_manager.py --clear                          # Clear all cache (with confirmation)
    python cache_manager.py --clear --dataset NAME           # Clear cache for specific dataset
    python cache_manager.py --rebuild                        # Clear and rebuild all cache
    python cache_manager.py --rebuild --dataset NAME         # Clear and rebuild specific dataset
"""

import argparse
import sys
from pathlib import Path
from etl.cache import ETLCache
from etl.raw_cache import RawDataCache


def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def show_info():
    """Show cache statistics and disk space"""
    cache = ETLCache()
    raw_cache = RawDataCache()

    cache_dir = Path("results/_cache")

    print("\n" + "=" * 70)
    print("CACHE INFORMATION")
    print("=" * 70)

    # Full cache info
    full_cache_files = list(cache_dir.glob("*.pkl")) + list(cache_dir.glob("*.pkl.gz"))
    full_cache_files = [f for f in full_cache_files if not f.name.startswith("raw_")]
    full_cache_size = sum(f.stat().st_size for f in full_cache_files) if full_cache_files else 0

    print(f"\nFull Cache:")
    print(f"  Location: {cache_dir}")
    print(f"  Files: {len(full_cache_files)}")
    print(f"  Total size: {format_size(full_cache_size)}")

    # Raw cache info
    raw_cache_files = list(cache_dir.glob("raw_*.pkl")) + list(cache_dir.glob("raw_*.pkl.gz"))
    raw_cache_size = sum(f.stat().st_size for f in raw_cache_files) if raw_cache_files else 0

    print(f"\nRaw Cache:")
    print(f"  Files: {len(raw_cache_files)}")
    print(f"  Total size: {format_size(raw_cache_size)}")

    # Total
    total_size = full_cache_size + raw_cache_size
    print(f"\nTotal Cache Size: {format_size(total_size)}")

    # Disk space
    import shutil
    stat = shutil.disk_usage(cache_dir)
    print(f"\nDisk Space:")
    print(f"  Total: {format_size(stat.total)}")
    print(f"  Used: {format_size(stat.used)} ({stat.used / stat.total * 100:.1f}%)")
    print(f"  Free: {format_size(stat.free)} ({stat.free / stat.total * 100:.1f}%)")

    if stat.free < 100 * 1024 * 1024:  # Less than 100 MB
        print(f"\n[!] WARNING: Low disk space! ({format_size(stat.free)} available)")
        print(f"   Consider clearing cache or freeing up disk space.")
    elif stat.free < 1024 * 1024 * 1024:  # Less than 1 GB
        print(f"\n[!] CAUTION: Disk space below 1 GB ({format_size(stat.free)} available)")

    print("\n" + "=" * 70 + "\n")


def list_cached():
    """List all cached datasets"""
    cache = ETLCache()
    raw_cache = RawDataCache()

    print("\n" + "=" * 70)
    print("CACHED DATASETS")
    print("=" * 70)

    print("\nFull Cache:")
    if cache.metadata:
        for key, info in sorted(cache.metadata.items()):
            print(f"  [+] {info.get('file_name', key)}")
    else:
        print("  (none)")

    print("\nRaw Cache:")
    if raw_cache.metadata:
        for key, info in sorted(raw_cache.metadata.items()):
            db_count = info.get('num_dbs', 'unknown')
            print(f"  [+] {info.get('dataset_name', key)} ({db_count} databases)")
    else:
        print("  (none)")

    print("\n" + "=" * 70 + "\n")


def clear_all():
    """Clear all cache"""
    cache = ETLCache()
    raw_cache = RawDataCache()

    print("\n" + "=" * 70)
    print("CLEARING ALL CACHE")
    print("=" * 70 + "\n")

    cache.clear()
    raw_cache.clear()

    print("[OK] All cache cleared successfully\n")


def clear_dataset(dataset_name):
    """Clear cache for specific dataset"""
    cache = ETLCache()
    raw_cache = RawDataCache()

    print("\n" + "=" * 70)
    print(f"CLEARING CACHE FOR: {dataset_name}")
    print("=" * 70 + "\n")

    cache.clear(dataset_name)
    raw_cache.clear(dataset_name)

    print(f"[OK] Cache cleared for {dataset_name}\n")


def build_cache_for_dataset(zip_path, results_dir):
    """Build cache for a specific dataset"""
    from etl.loader import etl

    print(f"\n  Processing: {zip_path.name}")
    print(f"  " + "-" * 60)

    try:
        # Force refresh to rebuild cache
        db = etl([zip_path], results_dir, use_cache=True, force_refresh=True)

        if "workers_report" in db:
            worker_count = db["workers_report"].get("worker_count", 0)
            print(f"  [OK] Successfully built cache ({worker_count} workers)")
            return True
        else:
            print(f"  [!] Warning: Cache built but no workers_report")
            return True

    except Exception as e:
        print(f"  [ERROR] Error building cache: {e}")
        return False


def build_all_cache(results_dir):
    """Build cache for all datasets"""
    from etl.loader import etl
    import builtins

    # Mock input to avoid interactive prompts
    original_input = builtins.input
    def mock_input(prompt=""):
        return ""
    builtins.input = mock_input

    zip_paths = sorted(results_dir.glob('*.zip'))

    if len(zip_paths) == 0:
        print("[ERROR] No ZIP files found in results directory")
        return

    print("\n" + "=" * 70)
    print(f"BUILDING CACHE FOR ALL DATASETS ({len(zip_paths)} files)")
    print("=" * 70)

    success_count = 0
    error_count = 0

    for i, zip_path in enumerate(zip_paths, 1):
        print(f"\n[{i}/{len(zip_paths)}]", end=" ")

        if build_cache_for_dataset(zip_path, results_dir):
            success_count += 1
        else:
            error_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total: {len(zip_paths)}")
    print(f"  [OK] Success: {success_count}")
    if error_count > 0:
        print(f"  [ERROR] Errors: {error_count}")
    print()

    # Restore original input
    builtins.input = original_input


def build_dataset_cache(dataset_name, results_dir):
    """Build cache for a specific dataset"""
    import builtins

    # Mock input to avoid interactive prompts
    original_input = builtins.input
    def mock_input(prompt=""):
        return ""
    builtins.input = mock_input

    # Find the dataset
    zip_paths = list(results_dir.glob('*.zip'))
    target_path = None

    for path in zip_paths:
        if path.name == dataset_name or path.stem == dataset_name:
            target_path = path
            break

    if target_path is None:
        print(f"\n[ERROR] Dataset '{dataset_name}' not found in {results_dir}")
        print(f"\nAvailable datasets:")
        for path in sorted(zip_paths):
            print(f"  - {path.name}")
        return False

    print("\n" + "=" * 70)
    print(f"BUILDING CACHE FOR: {target_path.name}")
    print("=" * 70)

    result = build_cache_for_dataset(target_path, results_dir)

    print("=" * 70)
    if result:
        print("[OK] Cache built successfully")
    else:
        print("[ERROR] Failed to build cache")
    print()

    # Restore original input
    builtins.input = original_input

    return result


def rebuild_all():
    """Clear and rebuild all cache"""
    print("\n" + "=" * 70)
    print("REBUILD ALL CACHE")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Clear all existing cache")
    print("  2. Rebuild cache for all datasets")
    print()

    confirm = input("Are you sure you want to continue? (y/N): ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        return

    # Clear first
    clear_all()

    # Then rebuild
    results_dir = Path('results')
    build_all_cache(results_dir)


def rebuild_dataset(dataset_name):
    """Clear and rebuild cache for specific dataset"""
    print("\n" + "=" * 70)
    print(f"REBUILD CACHE FOR: {dataset_name}")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Clear existing cache for this dataset")
    print("  2. Rebuild cache from source ZIP")
    print()

    # Clear first
    clear_dataset(dataset_name)

    # Then rebuild
    results_dir = Path('results')
    build_dataset_cache(dataset_name, results_dir)


def main():
    parser = argparse.ArgumentParser(
        description="ETL Cache Manager - Manage, build and clear cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Information and listing
  python cache_manager.py --info                         # Show cache statistics
  python cache_manager.py --list                         # List all cached datasets
  
  # Building cache
  python cache_manager.py --build                        # Build cache for all datasets
  python cache_manager.py --build --dataset Coffee.zip   # Build cache for specific dataset
  
  # Clearing cache
  python cache_manager.py --clear                        # Clear all cache (with confirmation)
  python cache_manager.py --clear --dataset Coffee.zip   # Clear specific dataset cache
  
  # Rebuilding (clear + build)
  python cache_manager.py --rebuild                      # Rebuild all cache
  python cache_manager.py --rebuild --dataset Coffee.zip # Rebuild specific dataset
        """
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Show cache statistics and disk space'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all cached datasets'
    )

    parser.add_argument(
        '--build',
        action='store_true',
        help='Build cache (for all datasets or specific dataset with --dataset)'
    )

    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear cache (all or specific dataset with --dataset)'
    )

    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Clear and rebuild cache (all or specific dataset with --dataset)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        metavar='NAME',
        help='Specify dataset name (e.g., Coffee.zip or Coffee_0_false_0.zip)'
    )

    args = parser.parse_args()

    # Validate args
    action_count = sum([args.info, args.list, args.build, args.clear, args.rebuild])

    if action_count == 0:
        parser.print_help()
        sys.exit(0)

    if action_count > 1:
        print("Error: Please specify only one action (--info, --list, --build, --clear, or --rebuild)")
        sys.exit(1)

    # Execute actions
    try:
        if args.info:
            show_info()

        elif args.list:
            list_cached()

        elif args.build:
            results_dir = Path('results')
            if not results_dir.exists():
                print(f"Error: Results directory '{results_dir}' not found")
                sys.exit(1)

            if args.dataset:
                build_dataset_cache(args.dataset, results_dir)
            else:
                build_all_cache(results_dir)

        elif args.clear:
            if args.dataset:
                clear_dataset(args.dataset)
            else:
                confirm = input("Are you sure you want to clear ALL cache? (y/N): ")
                if confirm.lower() == 'y':
                    clear_all()
                else:
                    print("Cancelled.")

        elif args.rebuild:
            if args.dataset:
                rebuild_dataset(args.dataset)
            else:
                rebuild_all()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
