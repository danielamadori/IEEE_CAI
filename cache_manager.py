"""
Utilità per gestire tutte le cache del progetto
"""

import os
from pathlib import Path
import shutil


def clear_all_cache():
    """Cancella tutte le cache del progetto"""
    cache_dir = Path("results/_cache")

    if not cache_dir.exists():
        print("✓ Nessuna cache da cancellare")
        return

    print("Cancellazione cache in corso...")

    # Lista file prima di cancellare
    pkl_files = list(cache_dir.glob("*.pkl"))
    csv_files = list(cache_dir.glob("*.csv"))
    json_files = list(cache_dir.glob("*.json"))

    print(f"  - {len(pkl_files)} file .pkl (reasons_analysis cache)")
    print(f"  - {len(csv_files)} file .csv (models_analysis cache)")
    print(f"  - {len(json_files)} file .json (metadata)")

    # Cancella directory
    shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("✓ Tutte le cache cancellate!")


def clear_reasons_cache():
    """Cancella solo cache di reasons_analysis.ipynb"""
    from etl.loader import clear_cache, clear_raw_cache

    print("Cancellazione cache reasons_analysis...")
    clear_cache()
    clear_raw_cache()
    print("✓ Cache reasons_analysis cancellata!")


def clear_models_cache():
    """Cancella solo cache di models_analysis.ipynb"""
    cache_dir = Path("results/_cache")

    files_removed = []

    counts_cache = cache_dir / "_counts_cache.csv"
    if counts_cache.exists():
        counts_cache.unlink()
        files_removed.append("_counts_cache.csv")

    meta_cache = cache_dir / "redis_counts_meta.json"
    if meta_cache.exists():
        meta_cache.unlink()
        files_removed.append("redis_counts_meta.json")

    summary_file = Path("results/redis_reason_counts.csv")
    if summary_file.exists():
        summary_file.unlink()
        files_removed.append("redis_reason_counts.csv")

    if files_removed:
        print(f"✓ Cancellati: {', '.join(files_removed)}")
    else:
        print("✓ Nessuna cache models_analysis da cancellare")


def list_all_cache():
    """Elenca tutte le cache disponibili"""
    from etl.loader import list_cache, list_raw_cache

    print("=" * 70)
    print("CACHE DISPONIBILI")
    print("=" * 70)
    print()

    # Reasons analysis cache
    print("1. REASONS_ANALYSIS CACHE (Full)")
    full_cache = list_cache()
    if full_cache:
        for dataset, info in full_cache.items():
            print(f"   • {info['file_name']}")
    else:
        print("   (nessuna)")
    print()

    print("2. REASONS_ANALYSIS CACHE (Raw)")
    raw_cache = list_raw_cache()
    if raw_cache:
        for dataset, info in raw_cache.items():
            print(f"   • {info['dataset_name']}: {info['num_dbs']} databases")
    else:
        print("   (nessuna)")
    print()

    # Models analysis cache
    print("3. MODELS_ANALYSIS CACHE")
    cache_dir = Path("results/_cache")
    models_files = []

    if (cache_dir / "_counts_cache.csv").exists():
        models_files.append("_counts_cache.csv")
    if (cache_dir / "redis_counts_meta.json").exists():
        models_files.append("redis_counts_meta.json")
    if Path("results/redis_reason_counts.csv").exists():
        models_files.append("redis_reason_counts.csv")

    if models_files:
        for f in models_files:
            print(f"   • {f}")
    else:
        print("   (nessuna)")
    print()

    # Statistiche totali
    total_pkl = len(list(cache_dir.glob("*.pkl"))) if cache_dir.exists() else 0
    total_csv = len(list(cache_dir.glob("*.csv"))) if cache_dir.exists() else 0
    total_json = len(list(cache_dir.glob("*.json"))) if cache_dir.exists() else 0

    print("=" * 70)
    print(f"TOTALE: {total_pkl} pkl, {total_csv} csv, {total_json} json")
    print("=" * 70)


def get_cache_size():
    """Calcola dimensione totale cache"""
    cache_dir = Path("results/_cache")

    if not cache_dir.exists():
        return 0

    total_size = 0
    for file in cache_dir.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size

    # Converti in MB
    size_mb = total_size / (1024 * 1024)
    return size_mb


def show_cache_info():
    """Mostra informazioni dettagliate sulla cache"""
    print("=" * 70)
    print("INFORMAZIONI CACHE")
    print("=" * 70)
    print()

    cache_dir = Path("results/_cache")

    if not cache_dir.exists():
        print("✗ Directory cache non esiste")
        return

    print(f"📁 Directory: {cache_dir.absolute()}")
    print()

    # Conta file per tipo
    pkl_files = list(cache_dir.glob("*.pkl"))
    csv_files = list(cache_dir.glob("*.csv"))
    json_files = list(cache_dir.glob("*.json"))

    print("File per tipo:")
    print(f"  • PKL:  {len(pkl_files)} file")
    print(f"  • CSV:  {len(csv_files)} file")
    print(f"  • JSON: {len(json_files)} file")
    print()

    # Dimensione totale
    size_mb = get_cache_size()
    print(f"💾 Dimensione totale: {size_mb:.2f} MB")
    print()

    # File più grandi
    all_files = list(cache_dir.rglob("*"))
    file_sizes = [(f, f.stat().st_size) for f in all_files if f.is_file()]
    file_sizes.sort(key=lambda x: x[1], reverse=True)

    if file_sizes:
        print("File più grandi:")
        for f, size in file_sizes[:5]:
            size_mb = size / (1024 * 1024)
            print(f"  • {f.name}: {size_mb:.2f} MB")
    print()


def interactive_menu():
    """Menu interattivo per gestire cache"""
    while True:
        print()
        print("=" * 70)
        print("GESTIONE CACHE - Menu Interattivo")
        print("=" * 70)
        print()
        print("1. Mostra cache disponibili")
        print("2. Mostra informazioni cache")
        print("3. Cancella TUTTE le cache")
        print("4. Cancella solo cache reasons_analysis")
        print("5. Cancella solo cache models_analysis")
        print("0. Esci")
        print()

        choice = input("Scelta: ").strip()

        if choice == "1":
            list_all_cache()
        elif choice == "2":
            show_cache_info()
        elif choice == "3":
            confirm = input("⚠️  Cancellare TUTTE le cache? (y/N): ").strip().lower()
            if confirm == 'y':
                clear_all_cache()
            else:
                print("✗ Operazione annullata")
        elif choice == "4":
            clear_reasons_cache()
        elif choice == "5":
            clear_models_cache()
        elif choice == "0":
            print("\n👋 Arrivederci!")
            break
        else:
            print("✗ Scelta non valida")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            list_all_cache()
        elif command == "info":
            show_cache_info()
        elif command == "clear-all":
            clear_all_cache()
        elif command == "clear-reasons":
            clear_reasons_cache()
        elif command == "clear-models":
            clear_models_cache()
        elif command == "size":
            size = get_cache_size()
            print(f"Cache size: {size:.2f} MB")
        else:
            print("Comandi disponibili:")
            print("  list          - Elenca cache disponibili")
            print("  info          - Mostra informazioni dettagliate")
            print("  clear-all     - Cancella tutte le cache")
            print("  clear-reasons - Cancella cache reasons_analysis")
            print("  clear-models  - Cancella cache models_analysis")
            print("  size          - Mostra dimensione cache")
            print()
            print("Senza parametri: menu interattivo")
    else:
        interactive_menu()

