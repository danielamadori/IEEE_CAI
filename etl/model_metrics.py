"""
Helpers to inspect model metrics stored inside DB0.

These utilities are shared by notebooks and table builders that need the
accuracy/CV scores extracted from the RF optimization block.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def _to_float(value: Any | None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_model_accuracy_from_db(db: Dict[str, Any] | None,
                                   dataset_name: str | None = None) -> Dict[str, Any]:
    """
    Extract key RF optimization metrics (test accuracy, CV score, best params)
    from the DB0 structure loaded by the ETL.
    """
    result = {
        "dataset": dataset_name,
        "test_accuracy": None,
        "cv_score": None,
        "best_params": None,
        "raw_metrics": {},
    }

    if not isinstance(db, Mapping):
        return result

    data_block = db.get("data")
    if not isinstance(data_block, Mapping):
        return result

    rf_entry = data_block.get("RF_OPTIMIZATION_RESULTS")
    if not isinstance(rf_entry, Mapping):
        return result

    rf_metrics = rf_entry.get("value_json")
    if not isinstance(rf_metrics, Mapping):
        return result

    result["raw_metrics"] = dict(rf_metrics)
    result["test_accuracy"] = _to_float(rf_metrics.get("test_score"))
    result["cv_score"] = _to_float(rf_metrics.get("best_cv_score"))
    result["best_params"] = rf_metrics.get("best_params")
    return result


def list_rf_metric_names(db: Dict[str, Any] | None) -> list[str]:
    """Return the ordered list of metric keys stored in RF_OPTIMIZATION_RESULTS."""
    info = extract_model_accuracy_from_db(db)
    return sorted(info["raw_metrics"].keys()) if info["raw_metrics"] else []
