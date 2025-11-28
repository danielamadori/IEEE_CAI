"""Utilities to analyze drift results and expose notebook helpers.

This module centralizes the logic that used to live in
`scripts/analyze_results.py`, `drifts_results.py`, and the temporary helper
`_tmp_counts.py`.  It can be imported from notebooks (e.g.
`from drifts_results import compute_counts_from_results, load_analyzed_df, ...`)
and also executed as a script to generate CSV reports or diagnostics.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gc
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover - notebooks expect pandas to be present
    print("pandas non trovato: installalo con `pip install pandas`", file=sys.stderr)
    raise


DB_TO_CAT: Dict[int, str] = {
    0: "DATA",
    1: "CAN",
    2: "R",
    3: "NR",
    4: "CAR",
    5: "AR",
    6: "GP",
    7: "BP",
    8: "PR",
    9: "AP",
    10: "LOGS",
}
CAT_LIST = [DB_TO_CAT[i] for i in sorted(DB_TO_CAT)]

CATEGORY_FULL_NAMES: Dict[str, str] = {
    "DATA": "Data entries",
    "CAN": "Candidate reasons (positive samples)",
    "R": "Reasons (confirmed positive ICFs)",
    "NR": "Non-reasons (confirmed negative ICFs)",
    "CAR": "Candidate anti-reasons (replaces DD)",
    "AR": "Anti-reasons (replaces DDS)",
    "GP": "Good profiles (reasons)",
    "BP": "Bad profiles (non-reasons)",
    "PR": "Preferred reasons",
    "AP": "Anti-reason profiles",
    "LOGS": "Worker iteration logs",
}
DISPLAY_CATEGORIES = ["CAN", "R", "NR", "CAR", "AR", "GP", "BP", "PR", "AP"]
DISPLAY_NAMES = {
    "CAN": "Candidate",
    "R": "Reason",
    "NR": "Non-reason",
    "CAR": "Candidate Anti-reason",
    "AR": "Anti-reason",
    "GP": "Good profile",
    "BP": "Bad profile",
    "PR": "Preferred reason",
    "AP": "Anti-reason profile",
}
SUMMARY_EXTRA_COLUMNS = ["log_start_min", "log_end_max", "log_duration_seconds"]
DISPLAY_LABELS = {**DISPLAY_NAMES, "TOT": "Total"}
RE_DB = re.compile(r"redis_backup_db(\d+)\.json$")
LOG_TIMESTAMP_RE = re.compile(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+")


def _detect_db_index(filename: str) -> Optional[int]:
    match = RE_DB.search(filename)
    return int(match.group(1)) if match else None


def _extract_log_timestamps_from_entry(entry: Dict[str, Any]) -> List[datetime]:
    """Decode a DB10 entry and collect ISO8601 timestamps."""
    value = entry.get("value")
    raw_bytes: Optional[bytes] = None
    if isinstance(value, dict):
        data = value.get("data")
        if isinstance(data, str):
            try:
                raw_bytes = base64.b64decode(data)
            except Exception:
                raw_bytes = None
        elif isinstance(data, bytes):
            raw_bytes = data
    elif isinstance(value, str):
        raw_bytes = value.encode("utf-8", errors="ignore")
    elif isinstance(value, bytes):
        raw_bytes = value
    if not raw_bytes:
        return []
    matches = LOG_TIMESTAMP_RE.findall(raw_bytes)
    timestamps: List[datetime] = []
    for match in matches:
        try:
            timestamps.append(datetime.fromisoformat(match.decode("ascii")))
        except Exception:
            continue
    return timestamps


def _update_log_bounds(
    bounds: Tuple[Optional[datetime], Optional[datetime]],
    candidates: List[datetime],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Merge new timestamp values into running min/max."""
    if not candidates:
        return bounds
    current_min, current_max = bounds
    candidate_min = min(candidates)
    candidate_max = max(candidates)
    if current_min is None or candidate_min < current_min:
        current_min = candidate_min
    if current_max is None or candidate_max > current_max:
        current_max = candidate_max
    return current_min, current_max


def _count_sample_entries(entries: Sequence[Dict[str, Any]]) -> int:
    """Count DATA keys matching the expected sample_* pattern (excluding *_meta)."""
    total = 0
    for entry in entries:
        key_b64 = entry.get("key")
        if not isinstance(key_b64, str):
            continue
        try:
            decoded = base64.b64decode(key_b64)
        except (binascii.Error, ValueError, TypeError):
            continue
        key_text = decoded.decode("utf-8", errors="ignore")
        if key_text.startswith("sample_") and not key_text.endswith("_meta"):
            total += 1
    return total


def _count_in_dir(ds_dir: Path, verbose: bool = False) -> Optional[Dict[str, Any]]:
    agg: Dict[str, Any] = {cat: 0 for cat in DB_TO_CAT.values()}
    agg["selected_sample"] = 0
    log_bounds: Tuple[Optional[datetime], Optional[datetime]] = (None, None)
    any_found = False
    for entry in ds_dir.rglob("redis_backup_db*.json"):
        db_index = _detect_db_index(entry.name)
        if db_index is None:
            continue

        data = None
        entries_list = None

        try:
            data = json.loads(entry.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            if verbose:
                print(f"?? Errore leggendo JSON: {entry}")
            continue

        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            cat = DB_TO_CAT.get(db_index)
            if cat:
                entries_list = data["entries"]
                count = len(entries_list)
                agg[cat] += count

                if cat == "DATA":
                    agg["selected_sample"] += _count_sample_entries(entries_list)
                elif cat == "LOGS":
                    # Estrai timestamp da tutti i log
                    timestamps: List[datetime] = []
                    for entry_obj in entries_list:
                        if isinstance(entry_obj, dict):
                            timestamps.extend(_extract_log_timestamps_from_entry(entry_obj))
                    log_bounds = _update_log_bounds(log_bounds, timestamps)
                    # Libera memoria dei timestamps
                    del timestamps

                any_found = True

        # Libera memoria del data JSON dopo l'elaborazione
        del data, entries_list
        data = None
        entries_list = None
    if not any_found and verbose:
        print(f"Manifest not found in directory: {ds_dir}")
        return None
    log_start, log_end = log_bounds
    if log_start:
        agg["log_start_min"] = log_start.isoformat()
    if log_end:
        agg["log_end_max"] = log_end.isoformat()
    if log_start and log_end:
        agg["log_duration_seconds"] = (log_end - log_start).total_seconds()
    return agg


def _count_in_zip(zip_path: Path, verbose: bool = False) -> Optional[Dict[str, Any]]:
    agg: Dict[str, Any] = {cat: 0 for cat in DB_TO_CAT.values()}
    agg["selected_sample"] = 0
    log_bounds: Tuple[Optional[datetime], Optional[datetime]] = (None, None)
    any_found = False
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            namelist = archive.namelist()
            for name in namelist:
                db_index = _detect_db_index(name)
                if db_index is None:
                    continue

                raw = None
                text = None
                data = None
                entries_list = None

                try:
                    raw = archive.read(name)
                    try:
                        text = raw.decode("utf-8")
                    except Exception:
                        text = raw.decode("latin-1", errors="ignore")
                    # Libera raw dopo decodifica
                    del raw
                    raw = None

                    data = json.loads(text)
                    # Libera text dopo parsing JSON
                    del text
                    text = None
                except Exception:
                    if verbose:
                        print(f"?? Errore leggendo {name} in {zip_path}")
                    # Libera eventuali riferimenti
                    del raw, text, data
                    continue

                if isinstance(data, dict) and isinstance(data.get("entries"), list):
                    cat = DB_TO_CAT.get(db_index)
                    if cat:
                        entries_list = data["entries"]
                        count = len(entries_list)
                        agg[cat] += count

                        if cat == "DATA":
                            agg["selected_sample"] += _count_sample_entries(entries_list)
                        elif cat == "LOGS":
                            # Estrai timestamp da tutti i log per Worker start/end/span
                            timestamps: List[datetime] = []
                            for entry_obj in entries_list:
                                if isinstance(entry_obj, dict):
                                    timestamps.extend(
                                        _extract_log_timestamps_from_entry(entry_obj)
                                    )
                            log_bounds = _update_log_bounds(log_bounds, timestamps)
                            # Libera memoria dei timestamps
                            del timestamps

                        any_found = True

                # Libera memoria del data JSON dopo l'elaborazione
                del data, entries_list
                data = None
                entries_list = None
    except zipfile.BadZipFile:
        if verbose:
            print(f"?? Corrupted zip: {zip_path}")
        return None
    except Exception as exc:
        if verbose:
            print(f"?? Errore leggendo zip: {zip_path} -> {exc}")
        return None
    if not any_found and verbose:
        print(f"Manifest not found in {zip_path}")
        return None
    log_start, log_end = log_bounds
    if log_start:
        agg["log_start_min"] = log_start.isoformat()
    if log_end:
        agg["log_end_max"] = log_end.isoformat()
    if log_start and log_end:
        agg["log_duration_seconds"] = (log_end - log_start).total_seconds()
    return agg


def compute_counts_from_results(
    results_dir: Path,
    verbose: bool = False,
    cache_file: Optional[Path] = None,
    log_summary_file: Optional[Path] = None,
    selected_dataset: Optional[str] = None,
) -> pd.DataFrame:
    """Return a DataFrame with counts per dataset and category.

    Processes datasets one at a time with explicit garbage collection to minimize RAM usage.
    Writes results incrementally to cache_file if provided (one row per dataset).

    Args:
        results_dir: Directory containing result ZIP files
        verbose: Print progress messages
        cache_file: Optional path to write results incrementally (CSV format)
        log_summary_file: Optional path to write log summary incrementally (JSON format)
        selected_dataset: Optional dataset name to process (if None, processes all datasets)
    """
    rows: List[Dict[str, Any]] = []
    log_info: Dict[str, Dict[str, Any]] = {}
    base_columns = ["dataset", *DISPLAY_CATEGORIES, "TOT", "selected_sample"]

    if not results_dir.exists():
        if verbose:
            print(f"Results directory not found: {results_dir}")
        return pd.DataFrame(columns=base_columns)

    # Se cache_file esiste, carica i dataset già processati
    processed_datasets = set()
    if cache_file and cache_file.exists():
        try:
            existing_df = pd.read_csv(cache_file)
            if 'dataset' in existing_df.columns:
                processed_datasets = set(existing_df['dataset'].astype(str))
                rows = existing_df.to_dict('records')
                if verbose:
                    print(f"Resuming from cache: {len(processed_datasets)} datasets already processed")
        except Exception as e:
            if verbose:
                print(f"Could not load existing cache: {e}")

    # Carica log summary esistente
    if log_summary_file and log_summary_file.exists():
        try:
            log_info = json.loads(log_summary_file.read_text(encoding='utf-8'))
        except Exception:
            pass

    # Conta totale per progresso
    entries_list = sorted(results_dir.iterdir())

    # Se selected_dataset è specificato, filtra solo quel dataset
    if selected_dataset:
        entries_list = [
            entry for entry in entries_list
            if (entry.is_dir() and entry.name.split("_")[0] == selected_dataset) or
               (entry.is_file() and entry.suffix.lower() == ".zip" and entry.stem.split("_")[0] == selected_dataset)
        ]

    total_entries = len(entries_list)
    processed_count = len(processed_datasets)
    new_count = 0

    # Inizializza cache file con header se non esiste
    if cache_file and not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=base_columns).to_csv(cache_file, index=False)

    for idx, entry in enumerate(entries_list, 1):
        dataset = None

        if entry.is_dir():
            dataset = entry.name.split("_")[0] if "_" in entry.name else entry.name
        elif entry.is_file() and entry.suffix.lower() == ".zip":
            dataset = entry.stem.split("_")[0] if "_" in entry.stem else entry.stem
        else:
            continue

        # Salta se già processato
        if dataset in processed_datasets:
            continue

        if verbose and total_entries > 10:
            print(f"Processing {idx}/{total_entries}: {entry.name}...", end='\r')

        agg = None
        if entry.is_dir():
            agg = _count_in_dir(entry, verbose=verbose)
        elif entry.is_file() and entry.suffix.lower() == ".zip":
            agg = _count_in_zip(entry, verbose=verbose)

        if agg and dataset:
            # Estrai extra prima di modificare agg
            extras = {
                key: agg.pop(key)
                for key in list(agg.keys())
                if key in SUMMARY_EXTRA_COLUMNS
            }
            counts = {cat: agg.get(cat, 0) for cat in DISPLAY_CATEGORIES}
            row = {"dataset": dataset, **counts}
            row["selected_sample"] = agg.get("selected_sample", 0)
            rows.append(row)

            if extras:
                log_info[dataset] = extras

            # SCRIVI INCREMENTALMENTE nel cache file dopo ogni dataset
            if cache_file:
                try:
                    row_df = pd.DataFrame([row])
                    row_df.to_csv(cache_file, mode='a', header=False, index=False)
                except Exception as e:
                    if verbose:
                        print(f"\nWarning: Could not append to cache: {e}")

            # SCRIVI log summary incrementalmente
            if log_summary_file and extras:
                try:
                    log_summary_file.parent.mkdir(parents=True, exist_ok=True)
                    log_summary_file.write_text(
                        json.dumps(log_info, ensure_ascii=False, indent=2),
                        encoding='utf-8'
                    )
                except Exception as e:
                    if verbose:
                        print(f"\nWarning: Could not write log summary: {e}")

            processed_datasets.add(dataset)
            new_count += 1

            # Libera memoria esplicitamente dopo ogni dataset
            del agg, counts, row, extras

            # Garbage collection ogni 3 dataset per evitare accumulo
            if new_count % 3 == 0:
                gc.collect()

    if verbose and total_entries > 10:
        print(f"\nProcessed {new_count} new datasets ({len(processed_datasets)} total)")

    if not rows:
        return pd.DataFrame(columns=base_columns)
    df = pd.DataFrame(rows)
    for cat in DISPLAY_CATEGORIES:
        if cat not in df.columns:
            df[cat] = 0
    df[DISPLAY_CATEGORIES] = df[DISPLAY_CATEGORIES].fillna(0).astype(int)
    df["TOT"] = df[DISPLAY_CATEGORIES].sum(axis=1)
    if "selected_sample" not in df.columns:
        df["selected_sample"] = 0
    df = df[base_columns]
    df["selected_sample"] = df["selected_sample"].fillna(0).astype(int)
    if log_info:
        df.attrs["log_summary"] = log_info
    return df


def list_missing_manifests(results_dir: Path) -> List[str]:
    """Return paths to zips/dirs that do not contain redis manifests."""
    missing: List[str] = []
    if not results_dir.exists():
        return missing
    for entry in sorted(results_dir.iterdir()):
        if entry.is_dir():
            found = any(
                RE_DB.search(path.name)
                for path in entry.rglob("redis_backup_db*.json")
            )
            if not found:
                missing.append(str(entry))
        elif entry.is_file() and entry.suffix.lower() == ".zip":
            ok = False
            try:
                with zipfile.ZipFile(entry, "r") as archive:
                    for name in archive.namelist():
                        if not RE_DB.search(name):
                            continue
                        try:
                            raw = archive.read(name)
                            try:
                                text = raw.decode("utf-8")
                            except Exception:
                                text = raw.decode("latin-1", errors="ignore")
                            data = json.loads(text)
                            if isinstance(data, dict) and isinstance(data.get("entries"), list):
                                ok = True
                                break
                        except Exception:
                            continue
            except zipfile.BadZipFile:
                missing.append(str(entry))
                continue
            except Exception:
                missing.append(str(entry))
                continue
            if not ok:
                missing.append(str(entry))
    return missing


def load_analyzed_df(fr_csv: Path) -> pd.DataFrame:
    """Load an analyzed CSV, preferring rows marked as analyzed."""
    if not fr_csv.exists():
        return pd.DataFrame(columns=["dataset", "analyzed"])
    base = pd.read_csv(fr_csv)
    if "analyzed" in base.columns:
        analyzed = base[base["analyzed"] == "YES"].copy()
        if analyzed.empty:
            analyzed = base.copy()
    else:
        analyzed = base.copy()
        analyzed["analyzed"] = "YES"
    return analyzed


def cast_dataset_str(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the `dataset` column exists and is a string dtype."""
    attrs = dict(getattr(df, "attrs", {}))
    if "dataset" in df.columns:
        df = df.dropna(subset=["dataset"]).copy()
        df["dataset"] = df["dataset"].astype(str)
    if attrs:
        df.attrs.update(attrs)
    return df


def summarize_counts_from_df(
    df: pd.DataFrame,
) -> List[Tuple[str, int, Dict[str, int], Dict[str, Any]]]:
    """Return summary tuples (dataset, total, category counts, extras)."""
    if df.empty:
        return []
    log_info = df.attrs.get("log_summary", {}) if hasattr(df, "attrs") else {}
    summary: List[Tuple[str, int, Dict[str, int], Dict[str, Any]]] = []
    for _, row in df.sort_values("dataset").iterrows():
        dataset = str(row.get("dataset", "")).strip()
        counts = {cat: int(row.get(cat, 0) or 0) for cat in DISPLAY_CATEGORIES}
        extras = log_info.get(dataset, {}).copy()
        summary.append((dataset, sum(counts.values()), counts, extras))
    return summary


def summarize_counts(
    results_dir: Path, verbose: bool = False
) -> List[Tuple[str, int, Dict[str, int], Dict[str, Any]]]:
    """Convenience wrapper that computes counts and returns the summary list."""
    df = compute_counts_from_results(results_dir, verbose=verbose)
    return summarize_counts_from_df(df)


def print_counts_summary(
    results_dir: Path,
    df: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> None:
    """Print a concise summary of counts per dataset."""
    if df is None:
        df = compute_counts_from_results(results_dir, verbose=verbose)
    summary = summarize_counts_from_df(df)
    if not summary:
        print(f"No valid redis manifests found in {results_dir}")
        return
    for dataset, total, counts, extras in summary:
        labeled = {DISPLAY_LABELS.get(cat, cat): count for cat, count in counts.items()}
        labeled[DISPLAY_LABELS["TOT"]] = total
        if extras.get("log_start_min"):
            labeled["Worker start (min)"] = extras["log_start_min"]
        if extras.get("log_end_max"):
            labeled["Worker end (max)"] = extras["log_end_max"]
        if extras.get("log_duration_seconds") is not None:
            labeled["Worker span (s)"] = round(extras["log_duration_seconds"], 3)
        print(dataset, labeled)


def fix_notebook_calls(nb_path: Path = Path("models_analysis_enriched.ipynb")) -> int:
    """Fix notebook code cells replacing underscored helper calls."""
    if not nb_path.exists():
        print("Notebook not found:", nb_path)
        return 1
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("Failed to read notebook:", exc)
        return 1
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "\n".join(cell.get("source", []))
        src_new = src.replace(
            "_compute_counts_from_results(", "compute_counts_from_results("
        )
        src_new = src_new.replace("_load_analyzed_df(", "load_analyzed_df(")
        src_new = src_new.replace("_cast_dataset_str(", "cast_dataset_str(")
        if src_new != src:
            cell["source"] = [line + "\n" for line in src_new.splitlines()]
            changed = True
    if changed:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print("Fixed notebook calls in", nb_path)
    else:
        print("No changes needed in", nb_path)
    return 0


def patch_models_analysis_enriched(
    nb_path: Path = Path("models_analysis_enriched.ipynb"),
) -> int:
    """Patch the notebook to import helpers instead of redefining them inline."""
    if not nb_path.exists():
        print("Notebook not found:", nb_path)
        return 1
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("Failed to read notebook:", exc)
        return 1
    cells = nb.get("cells", [])
    import_cell_src = [
        "from pathlib import Path\n",
        "from drifts_results import compute_counts_from_results, load_analyzed_df, cast_dataset_str, CAT_LIST, DB_TO_CAT\n",
        "# Detect notebook base directory as robustly as possible\n",
        "try:\n",
        "    from IPython import get_ipython\n",
        "    ip = get_ipython()\n",
        "    BASE_DIR = Path(ip.run_line_magic('pwd', '')).resolve()\n",
        "except Exception:\n",
        "    BASE_DIR = Path.cwd().resolve()\n",
        "RESULTS_DIR = BASE_DIR / 'results'\n",
        "FR_CSV = BASE_DIR / 'forest_report.csv'\n",
    ]

    support_idx = None
    robust_idx = None
    for idx, cell in enumerate(cells):
        src = "\n".join(cell.get("source", []))
        if support_idx is None and "# === Funzioni di supporto ===" in src:
            support_idx = idx
        if robust_idx is None and "Robust handling when there are NO records" in src:
            robust_idx = idx

    if support_idx is None:
        insert_at = 0
        for idx, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                insert_at = idx
                break
        new_cell = {
            "cell_type": "code",
            "metadata": {},
            "source": import_cell_src,
            "outputs": [],
            "execution_count": None,
        }
        cells.insert(insert_at, new_cell)
        support_idx = insert_at
        print("Support functions marker not found; inserted import cell at index", insert_at)

    cells[support_idx]["cell_type"] = "code"
    cells[support_idx]["metadata"] = {}
    cells[support_idx]["source"] = import_cell_src
    cells[support_idx]["outputs"] = []
    cells[support_idx]["execution_count"] = None
    print("Replaced support functions cell at index", support_idx)

    if robust_idx is not None:
        print("Found robust block at", robust_idx, "removing to end")
        del cells[robust_idx:]
    else:
        print("No robust duplicate block found; nothing else removed")

    nb["cells"] = cells
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Notebook patched successfully")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analizza la cartella results e conta i manifest redis backup",
    )
    parser.add_argument(
        "--results",
        "-r",
        default="results",
        help="Percorso alla cartella results",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="analyzed_counts_results_only.csv",
        help="File CSV di output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Stampa diagnostica dettagliata",
    )
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help="Stampa la lista degli zip/cartelle senza manifest",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Stampa a video i conteggi per dataset (simile a _tmp_counts.py)",
    )
    parser.add_argument(
        "--fix-notebook-calls",
        action="store_true",
        help="Correggi chiamate con underscore nel notebook models_analysis_enriched.ipynb",
    )
    parser.add_argument(
        "--patch-notebook",
        action="store_true",
        help="Patch notebook per importare le funzioni condivise",
    )
    parser.add_argument(
        "--notebook",
        default="models_analysis_enriched.ipynb",
        help="Percorso al notebook da modificare",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    results_dir = Path(args.results)
    nb_path = Path(args.notebook)

    if args.fix_notebook_calls:
        return fix_notebook_calls(nb_path)
    if args.patch_notebook:
        return patch_models_analysis_enriched(nb_path)

    if args.list_missing:
        missing = list_missing_manifests(results_dir)
        if missing:
            print("Missing or corrupted (no manifest):")
            for item in missing:
                print(" -", item)
        else:
            print("Nessun file mancante o corrotto rilevato in", results_dir)
        return 0

    df = compute_counts_from_results(results_dir, verbose=args.verbose)

    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path.resolve()} (rows: {len(df)})")

    if args.print_summary or args.verbose:
        print_counts_summary(results_dir, df=df, verbose=args.verbose)

    return 0


__all__ = [
    "CAT_LIST",
    "DB_TO_CAT",
    "CATEGORY_FULL_NAMES",
    "DISPLAY_CATEGORIES",
    "DISPLAY_NAMES",
    "DISPLAY_LABELS",
    "SUMMARY_EXTRA_COLUMNS",
    "compute_counts_from_results",
    "list_missing_manifests",
    "load_analyzed_df",
    "cast_dataset_str",
    "summarize_counts",
    "summarize_counts_from_df",
    "print_counts_summary",
    "fix_notebook_calls",
    "patch_models_analysis_enriched",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
