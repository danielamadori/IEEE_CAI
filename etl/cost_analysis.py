"""
Cost analysis module for analyzing reason types and their costs.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any


def analyze_costs(cost_records: List[Dict[str, Any]], reason_types: List[str]) -> pd.DataFrame:
    """
    Analyze and visualize cost data for different reason types.

    Parameters
    ----------
    cost_records : List[Dict[str, Any]]
        List of cost records with keys: 'reason_type', 'bitmap', 'sample_id', 'cost'
    reason_types : List[str]
        List of reason type names (e.g., ['reasons', 'non_reasons', 'anti_reasons'])

    Returns
    -------
    pd.DataFrame
        DataFrame containing all cost records
    """
    cost_df = pd.DataFrame(cost_records)

    if cost_df.empty:
        print("No cost data available.")
        return cost_df

    # ========== SUMMARY PER CATEGORIA ==========
    print("="*80)
    print("COST ANALYSIS BY CATEGORY")
    print("="*80)

    reason_summary = (
        cost_df.groupby("reason_type")["cost"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .rename(columns={"std": "std_dev"})
    )

    # Display summary table
    try:
        from IPython.display import display
        display(
            reason_summary
            .style.format({
                "mean": "{:.6f}",
                "std_dev": "{:.6f}",
                "min": "{:.6f}",
                "median": "{:.6f}",
                "max": "{:.6f}",
            })
        )
    except ImportError:
        print(reason_summary)

    # Bar plot con errori
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        reason_summary.index,
        reason_summary["mean"],
        yerr=reason_summary["std_dev"].fillna(0.0),
        capsize=4,
        color="#1f77b4",
    )
    ax.set_title("Mean cost by reason type")
    ax.set_xlabel("Reason type")
    ax.set_ylabel("Mean cost")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Box plot distribuzioni per categoria
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(
        [cost_df[cost_df["reason_type"] == r]["cost"] for r in reason_summary.index],
        tick_labels=reason_summary.index
    )
    ax.set_title("Cost distribution by reason type")
    ax.set_ylabel("Cost")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ========== ANALISI DETTAGLIATA PER OGNI CATEGORIA ==========
    for reason_type in reason_types:
        _analyze_reason_type(cost_df, reason_type)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)

    return cost_df


def _analyze_reason_type(cost_df: pd.DataFrame, reason_type: str) -> None:
    """
    Perform detailed analysis for a specific reason type.

    Parameters
    ----------
    cost_df : pd.DataFrame
        DataFrame containing cost records
    reason_type : str
        The reason type to analyze
    """
    print("\n" + "="*80)
    print(f"DETAILED ANALYSIS: {reason_type.upper()}")
    print("="*80)

    subset_df = cost_df[cost_df["reason_type"] == reason_type].copy()

    if subset_df.empty:
        print(f"No data for {reason_type}.")
        return

    # Statistiche per bitmap
    grouped_stats = (
        subset_df.groupby("bitmap")["cost"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .rename(columns={"std": "std_dev"})
        .sort_values("mean")
    )

    ordered_bitmaps = list(grouped_stats.index)
    bitmap_lookup = {bitmap: idx for idx, bitmap in enumerate(ordered_bitmaps, start=1)}
    subset_df["bitmap_index"] = subset_df["bitmap"].map(bitmap_lookup)

    bitmap_stats = grouped_stats.reset_index(drop=True)
    bitmap_stats.index = range(1, len(bitmap_stats) + 1)
    bitmap_stats.index.name = "bitmap_index"

    print(f"\nStatistics per bitmap for {reason_type}:")
    try:
        from IPython.display import display
        display(
            bitmap_stats
            .style.format({
                "mean": "{:.6f}",
                "std_dev": "{:.6f}",
                "min": "{:.6f}",
                "median": "{:.6f}",
                "max": "{:.6f}",
            })
        )
    except ImportError:
        print(bitmap_stats)

    # 1. Bar plot: Mean cost per bitmap
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        bitmap_stats.index,
        bitmap_stats["mean"],
        yerr=bitmap_stats["std_dev"].fillna(0.0),
        capsize=4,
        color="#ff7f0e",
    )
    ax.set_title(f"Mean cost per bitmap index for {reason_type}")
    ax.set_xlabel("Bitmap index")
    ax.set_ylabel("Mean cost")
    ax.set_xticks(bitmap_stats.index)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # 2. Histogram: Distribuzione complessiva dei costi
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(subset_df["cost"], bins=30, color="#2ca02c", alpha=0.8, edgecolor='black')
    ax.set_title(f"Cost distribution for {reason_type}")
    ax.set_xlabel("Cost")
    ax.set_ylabel("Frequency")
    ax.axvline(subset_df["cost"].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {subset_df["cost"].mean():.4f}')
    ax.axvline(subset_df["cost"].median(), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {subset_df["cost"].median():.4f}')
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # 3. Box plot: Distribuzione per ogni bitmap
    fig, ax = plt.subplots(figsize=(max(10, len(bitmap_stats) * 0.5), 6))
    costs_by_bitmap = [subset_df[subset_df["bitmap_index"] == idx]["cost"].values
                       for idx in bitmap_stats.index]
    bp = ax.boxplot(costs_by_bitmap, labels=bitmap_stats.index, patch_artist=True)

    # Colora i box
    for patch in bp['boxes']:
        patch.set_facecolor('#d62728')
        patch.set_alpha(0.6)

    ax.set_title(f"Cost distribution per bitmap for {reason_type}")
    ax.set_xlabel("Bitmap index")
    ax.set_ylabel("Cost")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45 if len(bitmap_stats) > 15 else 0)
    plt.tight_layout()
    plt.show()

    # 4. Violin plot: Distribuzione dettagliata per bitmap (top 20)
    if len(bitmap_stats) > 0:
        top_n = min(20, len(bitmap_stats))
        top_bitmaps = bitmap_stats.head(top_n).index

        fig, ax = plt.subplots(figsize=(max(10, top_n * 0.5), 6))
        costs_top = [subset_df[subset_df["bitmap_index"] == idx]["cost"].values
                     for idx in top_bitmaps]
        parts = ax.violinplot(costs_top, positions=range(len(top_bitmaps)),
                              showmeans=True, showmedians=True)

        ax.set_title(f"Cost distribution (violin plot) - Top {top_n} bitmaps for {reason_type}")
        ax.set_xlabel("Bitmap index")
        ax.set_ylabel("Cost")
        ax.set_xticks(range(len(top_bitmaps)))
        ax.set_xticklabels(top_bitmaps)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    # 5. Scatter plot: Cost vs Sample per ogni bitmap
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx in bitmap_stats.index[:10]:  # Plot primi 10 bitmap
        bitmap_data = subset_df[subset_df["bitmap_index"] == idx]
        ax.scatter(
            [idx] * len(bitmap_data),
            bitmap_data["cost"],
            alpha=0.6,
            s=50,
            label=f"Bitmap {idx}" if idx <= 5 else None
        )

    ax.set_title(f"Cost scatter plot per bitmap for {reason_type}")
    ax.set_xlabel("Bitmap index")
    ax.set_ylabel("Cost")
    if len(bitmap_stats) <= 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # 6. Heatmap: Bitmap (righe) vs Sample (colonne)
    pivot_df = subset_df.pivot_table(index="bitmap_index", columns="sample_id", values="cost")
    pivot_df = pivot_df.reindex(bitmap_stats.index, axis=0)
    pivot_df = pivot_df.sort_index(axis=1)

    fig, ax = plt.subplots(
        figsize=(max(6, 0.75 * len(pivot_df.columns) + 2),
                max(4, 0.35 * len(bitmap_stats.index) + 1)),
    )
    im = ax.imshow(pivot_df.values, aspect="auto", cmap="viridis", interpolation='nearest')
    ax.set_title(f"Cost heatmap for {reason_type} (Bitmap × Sample)")
    ax.set_xlabel("Sample ID")
    ax.set_ylabel("Bitmap index")
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=8)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cost")
    plt.tight_layout()
    plt.show()

    # 7. Statistiche descrittive addizionali
    print(f"\nAdditional statistics for {reason_type}:")
    print(f"  Total ICFs: {len(bitmap_stats)}")
    print(f"  Total samples: {subset_df['sample_id'].nunique()}")
    print(f"  Total combinations: {len(subset_df)}")
    print(f"  Cost range: [{subset_df['cost'].min():.6f}, {subset_df['cost'].max():.6f}]")
    print(f"  Cost std dev: {subset_df['cost'].std():.6f}")
    print(f"  Cost variance: {subset_df['cost'].var():.6f}")
    print(f"  Cost skewness: {subset_df['cost'].skew():.6f}")
    print(f"  Cost kurtosis: {subset_df['cost'].kurtosis():.6f}")

    # Percentili
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(subset_df['cost'], p)
        print(f"    {p}th: {val:.6f}")

