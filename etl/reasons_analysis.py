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


def calculate_robustness_per_bitmap(cost_df: 'pd.DataFrame', num_features: int = None) -> 'pd.DataFrame':
    """
    Calculate robustness for each bitmap according to IEEE CAI 2026 paper definition.

    Formula from paper Definition (Robustness):
        r(C, x) = 1 - max_{ICF in AR_{C,y}} cost_x(ICF) / |features|

    Where:
        - C is the classifier
        - x is the sample
        - y = C(x) is the classification
        - AR_{C,y} is the set of Anti Reasons for class y
        - cost_x(ICF) is the cost function for ICF applied to sample x
        - |features| is the total number of features

    For each bitmap (which represents an ICF), we calculate:
        - max_cost: max{cost(s, bitmap) : s in test_set}
        - robustness: 1 - max_cost / |features| (if num_features provided)

    Parameters
    ----------
    cost_df : pd.DataFrame
        DataFrame with columns: reason_type, bitmap_index, sample_id, cost
    num_features : int, optional
        Number of features for normalization. If None, robustness equals max_cost

    Returns
    -------
    pd.DataFrame
        Robustness statistics per bitmap with columns:
        reason_type, bitmap_index, max_cost, robustness, mean_cost, min_cost,
        std_cost, n_samples
    """
    import pandas as pd

    # Group by bitmap and calculate statistics
    bitmap_robustness = cost_df.groupby(['reason_type', 'bitmap_index']).agg({
        'cost': ['max', 'mean', 'min', 'std', 'count']
    }).reset_index()

    # Rename columns
    bitmap_robustness.columns = [
        'reason_type', 'bitmap_index', 'max_cost',
        'mean_cost', 'min_cost', 'std_cost', 'n_samples'
    ]

    # Calculate robustness according to paper: r(C,x) = 1 - max_cost / |features|
    if num_features is not None and num_features > 0:
        bitmap_robustness['robustness'] = 1.0 - (bitmap_robustness['max_cost'] / num_features)
    else:
        # If num_features not provided, robustness is just the max_cost
        bitmap_robustness['robustness'] = bitmap_robustness['max_cost']

    # Sort by max_cost descending (higher cost = lower robustness when normalized)
    bitmap_robustness = bitmap_robustness.sort_values('max_cost', ascending=False)

    return bitmap_robustness


def calculate_robustness_summary(robustness_dict: Dict) -> Dict[str, float]:
    """
    Extract robustness summary from the robustness dictionary.

    Parameters
    ----------
    robustness_dict : dict
        Dictionary with keys 'reasons', 'non_reasons', 'anti_reasons',
        each containing 'cost', 'sample', 'icf', 'bitmap'

    Returns
    -------
    dict
        Summary with reason_type -> robustness value
    """
    summary = {}
    for reason_type in ['reasons', 'non_reasons', 'anti_reasons']:
        if reason_type in robustness_dict and 'cost' in robustness_dict[reason_type]:
            summary[reason_type] = robustness_dict[reason_type]['cost']

    return summary


def get_most_robust_bitmaps(bitmap_robustness_df: 'pd.DataFrame', top_n: int = 10) -> 'pd.DataFrame':
    """
    Get the top N most robust bitmaps.

    Parameters
    ----------
    bitmap_robustness_df : pd.DataFrame
        DataFrame from calculate_robustness_per_bitmap
    top_n : int
        Number of top bitmaps to return

    Returns
    -------
    pd.DataFrame
        Top N most robust bitmaps
    """
    return bitmap_robustness_df.head(top_n)


def get_robustness_statistics_by_type(bitmap_robustness_df: 'pd.DataFrame') -> Dict[str, Dict[str, float]]:
    """
    Calculate robustness statistics for each reason type.

    Parameters
    ----------
    bitmap_robustness_df : pd.DataFrame
        DataFrame from calculate_robustness_per_bitmap

    Returns
    -------
    dict
        Nested dictionary with reason_type -> {statistic -> value}
    """
    import pandas as pd

    stats = {}
    for reason_type in ['reasons', 'non_reasons', 'anti_reasons']:
        subset = bitmap_robustness_df[bitmap_robustness_df['reason_type'] == reason_type]
        if len(subset) > 0:
            stats[reason_type] = {
                'count': len(subset),
                'max_cost': float(subset['max_cost'].max()),
                'mean_max_cost': float(subset['max_cost'].mean()),
                'min_max_cost': float(subset['max_cost'].min()),
                'max_robustness': float(subset['robustness'].max()) if 'robustness' in subset.columns else None,
                'mean_robustness': float(subset['robustness'].mean()) if 'robustness' in subset.columns else None,
                'min_robustness': float(subset['robustness'].min()) if 'robustness' in subset.columns else None,
                'std_robustness': float(subset['robustness'].std()) if 'robustness' in subset.columns else None,
                'median_robustness': float(subset['robustness'].median()) if 'robustness' in subset.columns else None
            }

    return stats


def calculate_sample_robustness(cost_df: 'pd.DataFrame', num_features: int, sample_id: str = None) -> Dict[str, float]:
    """
    Calculate robustness for a specific sample according to IEEE CAI 2026 paper.

    Formula from paper Definition (Robustness):
        r(C, x) = 1 - max_{ICF in AR_{C,y}} cost_x(ICF) / |features|

    Where AR_{C,y} is the set of Anti Reasons for the sample's classification y.

    Parameters
    ----------
    cost_df : pd.DataFrame
        DataFrame with costs for the sample(s)
    num_features : int
        Number of features for normalization (|features| in the formula)
    sample_id : str, optional
        Sample ID to filter. If None, uses all data

    Returns
    -------
    dict
        Dictionary with robustness values for each reason type.
        Keys: 'reasons', 'non_reasons', 'anti_reasons'
        Values: robustness score in [0, 1] or None if no data
    """
    if sample_id is not None:
        sample_costs = cost_df[cost_df['sample_id'] == sample_id]
    else:
        sample_costs = cost_df

    robustness = {}
    for reason_type in ['reasons', 'non_reasons', 'anti_reasons']:
        subset = sample_costs[sample_costs['reason_type'] == reason_type]
        if len(subset) > 0:
            max_cost = subset['cost'].max()
            # Apply the formula: r(C,x) = 1 - max_cost / |features|
            robustness[reason_type] = 1.0 - (max_cost / num_features)
        else:
            robustness[reason_type] = None

    return robustness

