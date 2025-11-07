"""
ETL functions for reasons analysis - data extraction and processing
"""
from typing import Dict, List, Tuple, Any


def extract_test_samples(db: Dict[str, Any]) -> Tuple[Dict, List, List, List]:
    """
    Extract test samples from database

    Parameters
    ----------
    db : dict
        Database dictionary

    Returns
    -------
    tuple
        (tests_sample, X_test, test_ids, feature_names)
    """
    training_set = db["data"]["TRAINING_SET"]["value_json"]
    feature_names = training_set["feature_names"]

    tests_sample = {}
    X_test = []
    test_ids = []

    prefix = "sample_"
    suffix = "_meta"

    for raw_k, v in db["data"].items():
        if isinstance(raw_k, str) and raw_k.startswith(prefix) and raw_k.endswith(suffix):
            sample_id = raw_k[len(prefix):-len(suffix)]
            meta = v["value_json"]

            tests_sample[sample_id] = {
                **meta,
                "features": meta["sample_dict"]
            }

            X_test.append(meta["sample_dict"])
            test_ids.append(sample_id)

    return tests_sample, X_test, test_ids, feature_names


def calculate_costs_for_reasons(db: Dict, tests_sample: Dict, sigmas_all: Dict) -> Tuple[Dict, List]:
    """
    Calculate costs for all reason types

    Parameters
    ----------
    db : dict
        Database with reasons, non_reasons, anti_reasons
    tests_sample : dict
        Test samples dictionary
    sigmas_all : dict
        Sigma values for all samples

    Returns
    -------
    tuple
        (robustness, cost_records)
    """
    from redis_helpers.icf import bitmap_to_icf
    from cost_function import cost_function

    eu = db["data"]['EU']['value_json']

    robustness = {}
    cost_records = []
    reason_types = ["reasons", "non_reasons", "anti_reasons"]

    for r in reason_types:
        robustness[r] = {}
        for bitmap_string in db[r].keys():
            icf = bitmap_to_icf(bitmap_string, eu)

            max_cost = None
            for sample_id, sample_data in tests_sample.items():
                sample_data.setdefault(r, {})
                sample_entry = sample_data[r].setdefault(bitmap_string, {})

                sample_entry["icf"] = icf
                cost = cost_function(
                    sample=sample_data["features"],
                    sigmas=sigmas_all[sample_id],
                    icf=icf
                )
                sample_entry["cost"] = cost

                cost_records.append({
                    "reason_type": r,
                    "bitmap": bitmap_string,
                    "sample_id": sample_id,
                    "cost": cost,
                })

                if max_cost is None or max_cost < cost:
                    max_cost = cost
                    robustness[r]["cost"] = cost
                    robustness[r]["sample"] = sample_data["features"]
                    robustness[r]["icf"] = icf
                    robustness[r]["bitmap"] = bitmap_string

    return robustness, cost_records


def prepare_cost_dataframe(tests_sample: Dict) -> 'pd.DataFrame':
    """
    Prepare cost DataFrame from tests_sample structure

    Parameters
    ----------
    tests_sample : dict
        Test samples with cost data

    Returns
    -------
    pd.DataFrame
        Cost data in tabular format
    """
    import pandas as pd

    cost_data = []
    for reason_type in ['reasons', 'non_reasons', 'anti_reasons']:
        for sample_id, sample_data in tests_sample.items():
            if reason_type not in sample_data:
                continue

            for bitmap_string, reason_data in sample_data[reason_type].items():
                cost_data.append({
                    'sample_id': sample_id,
                    'bitmap_index': bitmap_string,
                    'reason_type': reason_type,
                    'cost': reason_data['cost'],
                    'icf': reason_data.get('icf', {})
                })

    return pd.DataFrame(cost_data)

