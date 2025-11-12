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


def calculate_all_samples_robustness(cost_df: 'pd.DataFrame', num_features: int,
                                     tests_sample: Dict, test_ids: List[str],
                                     verbose: bool = True) -> 'pd.DataFrame':
    """
    Calculate robustness for ALL test samples with prediction information.

    Parameters
    ----------
    cost_df : pd.DataFrame
        DataFrame with costs for all samples
    num_features : int
        Number of features for normalization
    tests_sample : dict
        Dictionary with sample metadata (predicted_label, actual_label, etc.)
    test_ids : list
        List of all test sample IDs
    verbose : bool
        If True, print progress indicators

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sample_id, robustness, predicted_label,
        actual_label, correct_prediction
    """
    import pandas as pd

    # Filter only anti_reasons for robustness calculation (as per paper)
    anti_reasons_df = cost_df[cost_df['reason_type'] == 'anti_reasons'].copy()

    sample_robustness_results = []

    if verbose:
        print(f"Processing {len(test_ids)} samples...")

    for idx, sample_id in enumerate(test_ids):
        if verbose and (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(test_ids)} samples...")

        rob = calculate_sample_robustness(anti_reasons_df, num_features, sample_id)
        robustness_value = rob.get('anti_reasons')

        # Extract prediction information from tests_sample
        sample_meta = tests_sample.get(sample_id, {})
        predicted_label = sample_meta.get('predicted_label', 'N/A')
        actual_label = sample_meta.get('actual_label', 'N/A')
        is_correct = sample_meta.get('prediction_correct', None)

        sample_robustness_results.append({
            'sample_id': sample_id,
            'robustness': robustness_value,
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'correct_prediction': is_correct
        })

    if verbose:
        print(f"✓ All {len(test_ids)} samples processed")

    return pd.DataFrame(sample_robustness_results)


def print_robustness_statistics(sample_robustness_df: 'pd.DataFrame') -> None:
    """
    Print comprehensive statistics for sample robustness.

    Parameters
    ----------
    sample_robustness_df : pd.DataFrame
        DataFrame from calculate_all_samples_robustness
    """
    import pandas as pd

    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS - SAMPLE ROBUSTNESS")
    print(f"{'='*80}")

    # Remove samples with no robustness value
    valid_robustness = sample_robustness_df['robustness'].dropna()

    if len(valid_robustness) == 0:
        print("\nWARNING: No valid robustness values found!")
        return

    print(f"\nTotal samples analyzed: {len(sample_robustness_df)}")
    print(f"Valid robustness values: {len(valid_robustness)}")
    print(f"Samples with N/A: {len(sample_robustness_df) - len(valid_robustness)}")

    # Prediction correctness statistics
    print(f"\n--- Prediction Correctness ---")
    correct_count = sample_robustness_df['correct_prediction'].sum()
    total_count = len(sample_robustness_df)
    incorrect_count = total_count - correct_count
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    print(f"Correct predictions:   {correct_count}/{total_count} ({accuracy:.1f}%)")
    print(f"Incorrect predictions: {incorrect_count}/{total_count} ({(100-accuracy):.1f}%)")

    # Descriptive statistics
    print(f"\n--- Descriptive Statistics ---")
    print(f"Mean:        {valid_robustness.mean():.6f}")
    print(f"Median:      {valid_robustness.median():.6f}")
    print(f"Std Dev:     {valid_robustness.std():.6f}")
    print(f"Min:         {valid_robustness.min():.6f}")
    print(f"Max:         {valid_robustness.max():.6f}")
    print(f"25th %ile:   {valid_robustness.quantile(0.25):.6f}")
    print(f"75th %ile:   {valid_robustness.quantile(0.75):.6f}")
    print(f"IQR:         {valid_robustness.quantile(0.75) - valid_robustness.quantile(0.25):.6f}")

    # Distribution analysis
    print(f"\n--- Distribution Analysis ---")
    low_robustness = (valid_robustness < 0.33).sum()
    medium_robustness = ((valid_robustness >= 0.33) & (valid_robustness < 0.67)).sum()
    high_robustness = (valid_robustness >= 0.67).sum()

    print(f"Low robustness (< 0.33):      {low_robustness} ({low_robustness/len(valid_robustness)*100:.1f}%)")
    print(f"Medium robustness (0.33-0.67): {medium_robustness} ({medium_robustness/len(valid_robustness)*100:.1f}%)")
    print(f"High robustness (>= 0.67):    {high_robustness} ({high_robustness/len(valid_robustness)*100:.1f}%)")

    # Robustness comparison: Correct vs Incorrect predictions
    print(f"\n--- Robustness by Prediction Correctness ---")
    correct_samples = sample_robustness_df[sample_robustness_df['correct_prediction'] == True]
    incorrect_samples = sample_robustness_df[sample_robustness_df['correct_prediction'] == False]

    correct_rob = None
    incorrect_rob = None

    if len(correct_samples) > 0:
        correct_rob = correct_samples['robustness'].dropna()
        if len(correct_rob) > 0:
            print(f"\nCorrect Predictions ({len(correct_samples)} samples):")
            print(f"  Mean robustness: {correct_rob.mean():.6f} ± {correct_rob.std():.6f}")
            print(f"  Median:          {correct_rob.median():.6f}")
            print(f"  Range:           [{correct_rob.min():.6f}, {correct_rob.max():.6f}]")

    if len(incorrect_samples) > 0:
        incorrect_rob = incorrect_samples['robustness'].dropna()
        if len(incorrect_rob) > 0:
            print(f"\nIncorrect Predictions ({len(incorrect_samples)} samples):")
            print(f"  Mean robustness: {incorrect_rob.mean():.6f} ± {incorrect_rob.std():.6f}")
            print(f"  Median:          {incorrect_rob.median():.6f}")
            print(f"  Range:           [{incorrect_rob.min():.6f}, {incorrect_rob.max():.6f}]")

            if correct_rob is not None and len(correct_rob) > 0:
                diff = correct_rob.mean() - incorrect_rob.mean()
                print(f"\n  Difference (Correct - Incorrect): {diff:.6f}")


def create_robustness_visualizations(sample_robustness_df: 'pd.DataFrame',
                                     dataset_name: str = 'Dataset'):
    """
    Create comprehensive visualizations for sample robustness.

    Parameters
    ----------
    sample_robustness_df : pd.DataFrame
        DataFrame from calculate_all_samples_robustness
    dataset_name : str
        Name of the dataset for plot titles

    Returns
    -------
    tuple
        (main_figure, quartile_figure) - Two plotly figures
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd

    valid_rob = sample_robustness_df['robustness'].dropna()

    if len(valid_rob) == 0:
        print("No valid robustness data to visualize")
        return None, None

    # Create subplots: Boxplot, Histogram, and Cumulative Distribution
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Boxplot - Robustness Distribution',
            'Histogram - Robustness Frequency',
            'Cumulative Distribution',
            'Sample Robustness Values'
        ),
        specs=[[{"type": "box"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # 1. Boxplot
    fig.add_trace(go.Box(
        y=valid_rob,
        name='Robustness',
        boxmean='sd',
        marker=dict(color='lightblue'),
        boxpoints='outliers'
    ), row=1, col=1)

    # 2. Histogram with distribution
    fig.add_trace(go.Histogram(
        x=valid_rob,
        nbinsx=30,
        name='Frequency',
        marker=dict(color='steelblue'),
        opacity=0.7
    ), row=1, col=2)

    # Add mean and median lines to histogram
    mean_val = valid_rob.mean()
    median_val = valid_rob.median()

    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean_val:.3f}",
                 annotation_position="top", row=1, col=2)
    fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                 annotation_text=f"Median: {median_val:.3f}",
                 annotation_position="bottom right", row=1, col=2)

    # 3. Cumulative Distribution
    sorted_rob = np.sort(valid_rob)
    cumulative = np.arange(1, len(sorted_rob) + 1) / len(sorted_rob)

    fig.add_trace(go.Scatter(
        x=sorted_rob,
        y=cumulative,
        mode='lines',
        name='CDF',
        line=dict(color='purple', width=2)
    ), row=2, col=1)

    # Add percentile markers
    for percentile in [0.25, 0.5, 0.75]:
        val = np.percentile(valid_rob, percentile * 100)
        fig.add_trace(go.Scatter(
            x=[val],
            y=[percentile],
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=[f'{percentile*100:.0f}%: {val:.3f}'],
            textposition="top center",
            showlegend=False
        ), row=2, col=1)

    # 4. Individual Sample Robustness Values (colored by prediction correctness)
    valid_samples_df = sample_robustness_df[sample_robustness_df['robustness'].notna()].copy()

    # Separate correct and incorrect predictions
    correct_mask = valid_samples_df['correct_prediction'] == True
    incorrect_mask = valid_samples_df['correct_prediction'] == False

    # Plot correct predictions
    if correct_mask.sum() > 0:
        correct_df = valid_samples_df[correct_mask]
        fig.add_trace(go.Scatter(
            x=list(range(len(correct_df))),
            y=correct_df['robustness'].values,
            mode='markers',
            name='Correct Predictions',
            marker=dict(size=5, color='green', symbol='circle'),
            text=[f'Sample {sid}<br>Predicted: {pred}<br>Actual: {act}'
                  for sid, pred, act in zip(correct_df['sample_id'],
                                           correct_df['predicted_label'],
                                           correct_df['actual_label'])],
            hovertemplate='%{text}<br>Robustness: %{y:.4f}<extra></extra>',
            legendgroup='correct'
        ), row=2, col=2)

    # Plot incorrect predictions
    if incorrect_mask.sum() > 0:
        incorrect_df = valid_samples_df[incorrect_mask]
        x_offset = correct_mask.sum()
        fig.add_trace(go.Scatter(
            x=list(range(x_offset, x_offset + len(incorrect_df))),
            y=incorrect_df['robustness'].values,
            mode='markers',
            name='Incorrect Predictions',
            marker=dict(size=5, color='red', symbol='x'),
            text=[f'Sample {sid}<br>Predicted: {pred}<br>Actual: {act}'
                  for sid, pred, act in zip(incorrect_df['sample_id'],
                                           incorrect_df['predicted_label'],
                                           incorrect_df['actual_label'])],
            hovertemplate='%{text}<br>Robustness: %{y:.4f}<extra></extra>',
            legendgroup='incorrect'
        ), row=2, col=2)

    # Add mean line
    fig.add_hline(y=mean_val, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean_val:.3f}",
                 row=2, col=2)

    # Update axes labels
    fig.update_xaxes(title_text="Robustness Value", row=1, col=2)
    fig.update_yaxes(title_text="Robustness", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Robustness Value", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)
    fig.update_xaxes(title_text="Sample Index", row=2, col=2)
    fig.update_yaxes(title_text="Robustness", row=2, col=2)

    fig.update_layout(
        title_text=f"Sample-Level Robustness Analysis - {dataset_name}<br><sub>Total Samples: {len(valid_rob)} | Mean: {mean_val:.4f} ± {valid_rob.std():.4f}</sub>",
        height=900,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.45,
            xanchor="left",
            x=0.52
        )
    )

    # Additional statistics plot: Robustness by quartile
    quartiles = pd.qcut(valid_rob, q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    quartile_counts = quartiles.value_counts().sort_index()

    fig2 = go.Figure(data=[
        go.Bar(
            x=quartile_counts.index.astype(str),
            y=quartile_counts.values,
            text=quartile_counts.values,
            textposition='auto',
            marker=dict(color=['red', 'orange', 'lightgreen', 'green'])
        )
    ])

    fig2.update_layout(
        title=f'Sample Distribution by Robustness Quartile - {dataset_name}',
        xaxis_title='Quartile',
        yaxis_title='Number of Samples',
        template='plotly_white',
        height=400
    )

    return fig, fig2


def visualize_all_time_series(tests_sample: Dict, test_ids: List[str],
                              feature_names: List[str], max_samples: int = 50):
    """
    Visualize all time series samples.

    Parameters
    ----------
    tests_sample : dict
        Dictionary with sample data
    test_ids : list
        List of test sample IDs
    feature_names : list
        List of feature names
    max_samples : int
        Maximum number of samples to plot

    Returns
    -------
    plotly.Figure
        Figure with all time series
    """
    import plotly.graph_objects as go
    import numpy as np

    samples_to_plot = test_ids[:min(max_samples, len(test_ids))]

    fig = go.Figure()

    for idx, sample_id in enumerate(samples_to_plot):
        sample_dict = tests_sample[sample_id]["features"]
        series = np.array([sample_dict[f] for f in feature_names])
        x_axis = np.arange(len(series))

        fig.add_trace(go.Scatter(
            x=x_axis,
            y=series,
            mode='lines',
            name=f'Sample {sample_id}',
            line=dict(width=1.5),
            opacity=0.7,
            showlegend=(len(samples_to_plot) <= 20)
        ))

    fig.update_layout(
        title=f'Time Series - All Test Samples ({len(samples_to_plot)} samples)',
        xaxis_title='Feature Index',
        yaxis_title='Value',
        template='plotly_white',
        height=600,
        showlegend=True if len(samples_to_plot) <= 20 else False,
        hovermode='closest'
    )

    return fig


def visualize_sample_with_icf(sample_id: str, tests_sample: Dict,
                              feature_names: List[str], reason_type: str = 'reasons'):
    """
    Visualize a single sample with its ICF constraints.

    Parameters
    ----------
    sample_id : str
        Sample ID to visualize
    tests_sample : dict
        Dictionary with sample data
    feature_names : list
        List of feature names
    reason_type : str
        Type of reason: 'reasons', 'non_reasons', or 'anti_reasons'

    Returns
    -------
    plotly.Figure or None
        Figure with sample and ICF, or None if no data
    """
    import plotly.graph_objects as go
    import numpy as np

    if reason_type not in tests_sample[sample_id] or len(tests_sample[sample_id][reason_type]) == 0:
        print(f"No {reason_type} found for this sample")
        return None

    # Get sample data
    sample_dict = tests_sample[sample_id]["features"]
    series = np.array([sample_dict[f] for f in feature_names])
    x_axis = np.arange(len(series))

    # Get first ICF
    first_bitmap = list(tests_sample[sample_id][reason_type].keys())[0]
    icf = tests_sample[sample_id][reason_type][first_bitmap]["icf"]
    cost = tests_sample[sample_id][reason_type][first_bitmap]["cost"]

    # Determine color based on reason type
    colors = {
        'reasons': ('green', 'darkgreen'),
        'non_reasons': ('orange', 'darkorange'),
        'anti_reasons': ('red', 'darkred')
    }
    color, dark_color = colors.get(reason_type, ('blue', 'darkblue'))

    # Create figure
    fig = go.Figure()

    # Add time series line
    fig.add_trace(go.Scatter(
        x=x_axis, y=series,
        mode='lines',
        name='Test Sample',
        line=dict(color='blue', width=2)
    ))

    # Add constraint intervals
    constrained_count = 0
    for idx, f in enumerate(feature_names):
        if f in icf:
            lower_bound, upper_bound = icf[f]

            # Skip if unbounded
            if np.isinf(lower_bound) and np.isinf(upper_bound):
                continue

            constrained_count += 1

            # Replace inf with reasonable bounds
            if np.isinf(lower_bound):
                lower_bound = series[idx] - 1.5
            if np.isinf(upper_bound):
                upper_bound = series[idx] + 1.5

            # Add vertical bar
            fig.add_trace(go.Scatter(
                x=[idx, idx],
                y=[lower_bound, upper_bound],
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(size=8, color=dark_color, symbol='line-ew-open'),
                name=f'Constraint {f}' if constrained_count == 1 else None,
                legendgroup='constraints',
                showlegend=(constrained_count == 1),
                hovertemplate=f'Feature: {f}<br>Interval: [{lower_bound:.3f}, {upper_bound:.3f}]<extra></extra>'
            ))

            # Add sample value marker
            fig.add_trace(go.Scatter(
                x=[idx],
                y=[series[idx]],
                mode='markers',
                marker=dict(size=6, color='blue', symbol='circle'),
                showlegend=False,
                hovertemplate=f'Sample value: {series[idx]:.3f}<extra></extra>'
            ))

    reason_label = reason_type.upper().replace('_', '-')
    fig.update_layout(
        title=f'Time Series with Maximal {reason_label} - Sample {sample_id}<br><sub>Cost: {cost:.4f} | Constrained features: {constrained_count}/{len(feature_names)}</sub>',
        xaxis_title='Feature Index',
        yaxis_title='Value',
        template='plotly_white',
        height=500,
        showlegend=True,
        hovermode='closest'
    )

    print(f"{reason_label} - Cost: {cost:.6f}")
    print(f"{reason_label} - Constrained features: {constrained_count}/{len(feature_names)}")

    return fig


def visualize_sample_comparison(sample_id: str, tests_sample: Dict, feature_names: List[str]):
    """
    Compare reason and anti-reason for a single sample.

    Parameters
    ----------
    sample_id : str
        Sample ID to visualize
    tests_sample : dict
        Dictionary with sample data
    feature_names : list
        List of feature names

    Returns
    -------
    plotly.Figure or None
        Combined figure with reason and anti-reason, or None if missing data
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np

    # Check if both reasons and anti_reasons exist
    if ("reasons" not in tests_sample[sample_id] or len(tests_sample[sample_id]["reasons"]) == 0 or
        "anti_reasons" not in tests_sample[sample_id] or len(tests_sample[sample_id]["anti_reasons"]) == 0):
        print("Need both reasons and anti_reasons for comparison")
        return None

    # Get sample data
    sample_dict = tests_sample[sample_id]["features"]
    series = np.array([sample_dict[f] for f in feature_names])
    x_axis = np.arange(len(series))

    # Get reason data
    first_reason_bitmap = list(tests_sample[sample_id]["reasons"].keys())[0]
    reason_icf = tests_sample[sample_id]["reasons"][first_reason_bitmap]["icf"]
    reason_cost = tests_sample[sample_id]["reasons"][first_reason_bitmap]["cost"]

    # Get anti-reason data
    first_ar_bitmap = list(tests_sample[sample_id]["anti_reasons"].keys())[0]
    ar_icf = tests_sample[sample_id]["anti_reasons"][first_ar_bitmap]["icf"]
    ar_cost = tests_sample[sample_id]["anti_reasons"][first_ar_bitmap]["cost"]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Maximal REASON - Sample {sample_id} (cost={reason_cost:.4f})',
            f'Maximal ANTI-REASON - Sample {sample_id} (cost={ar_cost:.4f})'
        ),
        vertical_spacing=0.12
    )

    # Top plot: Reason
    fig.add_trace(go.Scatter(
        x=x_axis, y=series,
        mode='lines',
        name='Test Sample',
        line=dict(color='blue', width=2),
        showlegend=True
    ), row=1, col=1)

    reason_constrained_count = 0
    for idx, f in enumerate(feature_names):
        if f in reason_icf:
            lower_bound, upper_bound = reason_icf[f]
            if not (np.isinf(lower_bound) and np.isinf(upper_bound)):
                reason_constrained_count += 1
                if np.isinf(lower_bound):
                    lower_bound = series[idx] - 1.5
                if np.isinf(upper_bound):
                    upper_bound = series[idx] + 1.5

                fig.add_trace(go.Scatter(
                    x=[idx, idx],
                    y=[lower_bound, upper_bound],
                    mode='lines+markers',
                    line=dict(color='green', width=4),
                    marker=dict(size=8, color='darkgreen', symbol='line-ew-open'),
                    name='Reason Constraints' if reason_constrained_count == 1 else None,
                    legendgroup='reason_constraints',
                    showlegend=(reason_constrained_count == 1),
                    hovertemplate=f'Feature: {f}<br>Interval: [{lower_bound:.3f}, {upper_bound:.3f}]<extra></extra>'
                ), row=1, col=1)

    # Bottom plot: Anti-Reason
    fig.add_trace(go.Scatter(
        x=x_axis, y=series,
        mode='lines',
        name='Test Sample',
        line=dict(color='blue', width=2),
        showlegend=False
    ), row=2, col=1)

    ar_constrained_count = 0
    for idx, f in enumerate(feature_names):
        if f in ar_icf:
            lower_bound, upper_bound = ar_icf[f]
            if not (np.isinf(lower_bound) and np.isinf(upper_bound)):
                ar_constrained_count += 1
                if np.isinf(lower_bound):
                    lower_bound = series[idx] - 1.5
                if np.isinf(upper_bound):
                    upper_bound = series[idx] + 1.5

                fig.add_trace(go.Scatter(
                    x=[idx, idx],
                    y=[lower_bound, upper_bound],
                    mode='lines+markers',
                    line=dict(color='red', width=4),
                    marker=dict(size=8, color='darkred', symbol='line-ew-open'),
                    name='Anti-Reason Constraints' if ar_constrained_count == 1 else None,
                    legendgroup='ar_constraints',
                    showlegend=(ar_constrained_count == 1),
                    hovertemplate=f'Feature: {f}<br>Interval: [{lower_bound:.3f}, {upper_bound:.3f}]<extra></extra>'
                ), row=2, col=1)

    fig.update_xaxes(title_text="Feature Index", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)

    fig.update_layout(
        height=900,
        template='plotly_white',
        showlegend=True,
        hovermode='closest'
    )

    print(f"\nReason: {reason_constrained_count} constrained features")
    print(f"Anti-Reason: {ar_constrained_count} constrained features")

    return fig





