#!/usr/bin/env python3
"""
Models Analysis Script

This script performs comprehensive analysis on machine learning model results.
It loads data from zip files, processes it through an ETL pipeline, and generates
various analysis tables and CSV outputs with model performance metrics.

The script integrates functionality from both models_analysis.py and reasons_analysis.ipynb:
- Model performance analysis and CSV generation
- Robustness analysis with cost calculations
- Anti-reasons analysis and visualizations
- Comprehensive reporting and statistical analysis

Usage:
    python models_analysis_script.py [--force-refresh] [--verbose] [--skip-robustness]

Output files:
    - combined_analyzed_results.csv: Main analysis results
    - combined_analyzed_results_updated.csv: Enhanced results with additional metrics
    - summary_results.csv: Summary statistics
    - analyzed_counts.csv: Detailed counts and metrics
    - results/sample_robustness.csv: Sample-level robustness analysis
    - results/anti_reasons_robustness.csv: Anti-reason ICF robustness analysis
    - results/*_robustness_report.txt: Comprehensive robustness report
    - figures/*.html: Interactive visualizations saved to dedicated figures folder
    - figures/*.pdf: High-quality PDF visualizations for publications
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_etl_pipeline(results_dir: Path, force_refresh: bool = False, verbose: bool = False):
    """
    Set up and run the ETL pipeline to load model analysis data.
    
    Args:
        results_dir (Path): Directory containing result zip files
        force_refresh (bool): Whether to force refresh the cache
        verbose (bool): Enable verbose logging
        
    Returns:
        Database object with loaded analysis data
    """
    try:
        from etl.loader import etl
        
        # Find all zip files in the results directory
        zip_paths = sorted(results_dir.glob("*.zip"))
        logger.info(f"Found {len(zip_paths)} zip files in {results_dir}")
        
        if not zip_paths:
            logger.warning(f"No zip files found in {results_dir}")
            return None
        
        # Load data through ETL pipeline
        db = etl(
            zip_paths,
            results_dir,
            use_cache=True,
            force_refresh=force_refresh,
            auto_select=True,
            load_only_db10=False,  # Load ALL databases from zip files
            verbose=verbose
        )
        
        logger.info("ETL pipeline completed successfully")
        return db
        
    except ImportError as e:
        logger.error(f"Failed to import ETL modules: {e}")
        raise
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        raise


def prepare_analysis(db, verbose: bool = False):
    """
    Prepare model analysis tables and context.
    
    Args:
        db: Database object from ETL pipeline
        verbose (bool): Enable verbose output
        
    Returns:
        Analysis context object with prepared tables
    """
    try:
        from etl.tables import (
            prepare_models_analysis,
            print_models_analysis_diagnostics,
        )
        
        # Prepare analysis context
        analysis_context = prepare_models_analysis(
            db=db, 
            verbose=verbose, 
            selected_dataset=None
        )
        
        # Print diagnostics if verbose
        if verbose:
            print_models_analysis_diagnostics(analysis_context)
            
        logger.info("Analysis preparation completed successfully")
        return analysis_context
        
    except ImportError as e:
        logger.error(f"Failed to import analysis modules: {e}")
        raise
    except Exception as e:
        logger.error(f"Analysis preparation failed: {e}")
        raise


def save_analysis_results(analysis_context, output_dir: Path = None):
    """
    Save analysis results to CSV files.
    
    Args:
        analysis_context: Analysis context object with prepared tables
        output_dir (Path): Output directory (defaults to current directory)
    """
    if output_dir is None:
        output_dir = Path(".")
    
    try:
        # Save combined analyzed results
        combined_df = analysis_context.combined_analyzed_styler.data
        combined_path = output_dir / "combined_analyzed_results.csv"
        combined_df.to_csv(combined_path)
        
        logger.info(f"Saved combined analysis results to: {combined_path}")
        logger.info(f"Shape: {combined_df.shape}")
        logger.info(f"Columns: {list(combined_df.columns)}")
        logger.info(f"Index: {list(combined_df.index)}")
        
        # Save first table summary
        summary_df = analysis_context.first_table.summary_styler.data
        summary_path = output_dir / "summary_results.csv"
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Saved summary results to: {summary_path}")
        logger.info(f"Shape: {summary_df.shape}")
        logger.info(f"Columns: {list(summary_df.columns)}")
        
        # Save analyzed counts
        counts_df = analysis_context.analyzed_counts_df
        counts_path = output_dir / "analyzed_counts.csv"
        counts_df.to_csv(counts_path, index=False)
        
        logger.info(f"Saved analyzed counts to: {counts_path}")
        logger.info(f"Shape: {counts_df.shape}")
        logger.info(f"Columns: {list(counts_df.columns)}")
        
        return {
            'combined_path': combined_path,
            'summary_path': summary_path,
            'counts_path': counts_path
        }
        
    except Exception as e:
        logger.error(f"Failed to save analysis results: {e}")
        raise

def enhance_combined_results(output_dir: Path = None):
    """
    Enhance the combined results by merging EU Features with standard deviation
    and adding additional metrics from other CSV files.
    
    Args:
        output_dir (Path): Output directory containing CSV files
        
    Returns:
        Enhanced DataFrame
    """
    if output_dir is None:
        output_dir = Path(".")
    
    try:
        # Read the current combined data
        combined_path = output_dir / "combined_analyzed_results.csv"
        combined_df = pd.read_csv(combined_path, index_col=0)
        
        # Combine Mean EU Features and EU Std into a single row
        if 'Mean EU Features' in combined_df.index and 'EU Std' in combined_df.index:
            logger.info("Combining Mean EU Features and EU Std...")
            
            mean_row = combined_df.loc['Mean EU Features']
            std_row = combined_df.loc['EU Std']
            
            # Create the combined row with format "mean (± std)"
            combined_row = pd.Series(index=combined_df.columns, dtype=object)
            
            for col in combined_df.columns:
                mean_val = mean_row[col]
                std_val = std_row[col]
                
                if pd.notna(mean_val) and pd.notna(std_val):
                    try:
                        mean_num = float(mean_val)
                        std_num = float(std_val)
                        combined_row[col] = f"{mean_num:.3f} (± {std_num:.3f})"
                    except (ValueError, TypeError):
                        combined_row[col] = str(mean_val)
                elif pd.notna(mean_val):
                    combined_row[col] = str(mean_val)
                else:
                    combined_row[col] = ""
            
            # Remove original rows and add combined row
            combined_df = combined_df.drop(['Mean EU Features', 'EU Std'])
            
            # Find insertion position
            insert_idx = 4
            if 'N Estimators' in combined_df.index:
                insert_idx = list(combined_df.index).index('N Estimators') + 1
            
            # Insert the new row
            combined_df_list = combined_df.index.tolist()
            combined_df_list.insert(insert_idx, 'Mean EU Features (± Std)')
            
            new_combined_df = pd.DataFrame(index=combined_df_list, columns=combined_df.columns)
            for idx in combined_df.index:
                new_combined_df.loc[idx] = combined_df.loc[idx]
            new_combined_df.loc['Mean EU Features (± Std)'] = combined_row
            
            combined_df = new_combined_df
            logger.info("Successfully combined Mean EU Features and EU Std")
        
        # Add data from summary and counts CSV files
        summary_path = output_dir / "summary_results.csv"
        counts_path = output_dir / "analyzed_counts.csv"
        
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            logger.info(f"Loaded summary_results.csv with {summary_df.shape[0]} datasets")
            
            # Add summary metrics
            summary_metrics = ['n_features', 'eu_complexity', 'eu_min', 'eu_max']
            for metric in summary_metrics:
                if metric in summary_df.columns:
                    new_row = pd.Series(index=combined_df.columns, dtype=object)
                    
                    for _, row in summary_df.iterrows():
                        dataset = str(row['dataset'])
                        if dataset in combined_df.columns:
                            value = row[metric]
                            if pd.notna(value):
                                new_row[dataset] = value
                    
                    # Format metric name
                    metric_names = {
                        'n_features': 'N Features',
                        'eu_complexity': 'EU Complexity',
                        'eu_min': 'EU Min',
                        'eu_max': 'EU Max'
                    }
                    metric_name = metric_names.get(metric, metric.replace('_', ' ').title())
                    combined_df.loc[metric_name] = new_row
                    logger.info(f"Added {metric_name} from summary_results.csv")
        
        if counts_path.exists():
            counts_df = pd.read_csv(counts_path)
            logger.info(f"Loaded analyzed_counts.csv with {counts_df.shape[0]} datasets")
            
            # Define metrics to exclude (already present or not relevant)
            exclude_metrics = {
                'dataset', 'Total time (s) max', 'Total time (s) mean',
                'ICF checks', 'Reason check iteration total',
                'IterGoodRatio', 'IterBadRatio', 'Early Stop Good total',
                'Early Stop from Good', 'Early Stop from Bad', 'Filtrered rate'
            }
            
            # Add relevant counts metrics
            for metric in counts_df.columns:
                if metric not in exclude_metrics:
                    new_row = pd.Series(index=combined_df.columns, dtype=object)
                    
                    for _, row in counts_df.iterrows():
                        dataset = str(row['dataset'])
                        if dataset in combined_df.columns and dataset != 'All workers':
                            value = row[metric]
                            if pd.notna(value):
                                new_row[dataset] = value
                    
                    # Add the row if it has any data
                    if new_row.notna().any():
                        metric_name = metric.replace('_', ' ').title()
                        combined_df.loc[metric_name] = new_row
                        logger.info(f"Added {metric_name} from analyzed_counts.csv")
        
        # Save enhanced results
        enhanced_path = output_dir / "combined_analyzed_results_updated.csv"
        combined_df.to_csv(enhanced_path)
        
        logger.info(f"\nFinal combined dataframe shape: {combined_df.shape}")
        logger.info(f"Total metrics: {len(combined_df.index)}")
        logger.info(f"Datasets: {len(combined_df.columns)}")
        logger.info(f"Saved enhanced results to: {enhanced_path}")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Failed to enhance combined results: {e}")
        raise

def visualization(anti_reasons_df, num_features: int, tests_sample: dict, sample_robustness_df, test_ids: list, feature_names: list, dataset_name: str, output_dir: Path = None):
    """
    Generate and save all visualizations to a dedicated figures folder.
    
    Args:
        anti_reasons_df: DataFrame with anti-reasons data
        num_features: Number of features
        tests_sample: Test sample data
        sample_robustness_df: Sample robustness DataFrame
        test_ids: List of test IDs
        feature_names: List of feature names
        dataset_name: Name of the dataset
        output_dir: Output directory for figures
    """
    if output_dir is None:
        output_dir = Path(".")
    
    # Create dedicated figures directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Calculate robustness per bitmap (ICF) - ONLY for Anti-Reasons
    from etl.reasons_analysis import calculate_robustness_per_bitmap
    print("\n" + "=" * 80)
    # Filter only anti-reasons
    print("ROBUSTNESS PER ICF (Anti-Reasons Only)")
    print("=" * 80)
    if len(anti_reasons_df) == 0:
        print("\nNo anti-reasons found in cost data.")
    else:
        # Calculate with normalization
        bitmap_robustness = calculate_robustness_per_bitmap(anti_reasons_df, num_features=num_features)

        print(f"\nTotal anti-reason ICFs analyzed: {len(bitmap_robustness)}")
        print(f"\nTop 10 ICFs by max_cost (hardest to reach):")
        print(bitmap_robustness[['max_cost', 'robustness', 'mean_cost', 'n_samples']].head(10).to_string(index=False))

        # Anti-reasons with lowest robustness (easiest to change classification)
        print(f"\n\nANTI-REASONS with LOWEST robustness (most vulnerable - easiest to perturb):")
        ar_lowest = bitmap_robustness.nsmallest(5, 'robustness')
        print(ar_lowest[['max_cost', 'robustness', 'mean_cost', 'n_samples']].to_string(index=False))

        # Save results (including bitmap_index for reference, but not displayed)
        bitmap_robustness.to_csv('results/anti_reasons_robustness.csv', index=False)
        print(f"\n\nResults saved to: results/anti_reasons_robustness.csv")
        
    # Statistical Visualizations for Sample Robustness
    from etl.reasons_analysis import create_robustness_visualizations

    if sample_robustness_df is not None and len(sample_robustness_df) > 0:
        print(f"\nGenerating robustness visualizations for {dataset_name}...")
        
        # Create visualizations using ETL function
        fig_main, fig_quartiles = create_robustness_visualizations(
            sample_robustness_df=sample_robustness_df,
            dataset_name=dataset_name
        )

        if fig_main is not None:
            main_path = figures_dir / f"{dataset_name}_robustness_main.html"
            fig_main.write_html(str(main_path))
            print(f"Main robustness visualization saved to: {main_path}")

        if fig_quartiles is not None:
            quartiles_path = figures_dir / f"{dataset_name}_robustness_quartiles.html"
            fig_quartiles.write_html(str(quartiles_path))
            print(f"Quartiles robustness visualization saved to: {quartiles_path}")
    else:
        print("sample_robustness_df not available. Skipping robustness visualizations.")
        
    # Visualize actual ICFs from the dataset
    from etl.reasons_analysis import visualize_all_time_series

    print(f"\nTotal test samples available: {len(test_ids)}")
    print(f"Number of features per sample: {len(feature_names)}")

    # Visualize all time series (max 50 samples)
    fig = visualize_all_time_series(tests_sample, test_ids, feature_names, max_samples=50)
    if fig is not None:
        all_series_path = figures_dir / f"{dataset_name}_all_time_series.html"
        fig.write_html(str(all_series_path))
        print(f"All time series visualization saved to: {all_series_path}")

    # Select first sample for detailed analysis
    sample_id = test_ids[0]

    print(f"\n{'='*80}")
    print(f"Detailed analysis will use Sample ID: {sample_id}")
    print(f"{'='*80}")
    
    # Plot: Time series with REASON (maximal reason from dataset)
    from etl.reasons_analysis import visualize_sample_with_icf

    fig = visualize_sample_with_icf(sample_id, tests_sample, feature_names, reason_type='reasons')
    if fig is not None:
        reason_path = figures_dir / f"{dataset_name}_sample_{sample_id}_reasons.html"
        fig.write_html(str(reason_path))
        print(f"Sample reasons visualization saved to: {reason_path}")
        
    # Plot: Time series comparison (smooth)
    from etl.reasons_analysis import visualize_sample_comparison_smooth

    fig = visualize_sample_comparison_smooth(sample_id, tests_sample, feature_names)
    if fig is not None:
        comparison_path = figures_dir / f"{dataset_name}_sample_{sample_id}_comparison_smooth.html"
        fig.write_html(str(comparison_path))
        print(f"Sample comparison (smooth) visualization saved to: {comparison_path}")
        
    # Plot: Time series with ANTI-REASON
    fig = visualize_sample_with_icf(sample_id, tests_sample, feature_names, reason_type='anti_reasons')
    if fig is not None:
        anti_reason_path = figures_dir / f"{dataset_name}_sample_{sample_id}_anti_reasons.html"
        fig.write_html(str(anti_reason_path))
        print(f"Sample anti-reasons visualization saved to: {anti_reason_path}")
        
    # Plot: Anti-reason corridor
    from etl.reasons_analysis import visualize_anti_reason_corridor

    fig = visualize_anti_reason_corridor(sample_id, tests_sample, feature_names)
    if fig is not None:
        # Save as HTML
        corridor_html_path = figures_dir / f"{dataset_name}_sample_{sample_id}_anti_reason_corridor.html"
        fig.write_html(str(corridor_html_path))
        print(f"Anti-reason corridor visualization saved to: {corridor_html_path}")
        
        # Save as PDF
        corridor_pdf_path = figures_dir / f"{dataset_name}_sample_{sample_id}_anti_reason_corridor.pdf"
        fig.write_image(str(corridor_pdf_path), format='pdf', width=1200, height=700, scale=1)
        print(f"Anti-reason corridor PDF saved to: {corridor_pdf_path}")
        
    print(f"\nAll visualizations saved to directory: {figures_dir}")
    return figures_dir
def perform_robustness_analysis(db, output_dir: Path = None, verbose: bool = False):
    """
    Perform comprehensive robustness analysis including cost calculations and visualizations.
    Integrates functionality from reasons_analysis.ipynb.
    
    Args:
        db: Database object from ETL pipeline
        output_dir (Path): Output directory for results
        verbose (bool): Enable verbose output
    """
    if output_dir is None:
        output_dir = Path(".")
    
    # Ensure results subdirectory exists
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    try:
        logger.info("Starting robustness analysis...")
        
        # Step 1: Extract test samples and calculate sigmas
        from cost_function import cal_sigmas
        from etl.reasons_analysis import extract_test_samples
        
        logger.info("Extracting test samples...")
        tests_sample, X_test, test_ids, feature_names = extract_test_samples(db)
        
        X_train = db["data"]["TRAINING_SET"]["value_json"]["X_train"]
        sigmas_all = cal_sigmas(X_train, X_test, feature_names, test_ids=test_ids)
        
        logger.info(f"Extracted {len(test_ids)} test samples with {len(feature_names)} features")
        
        # Step 2: Calculate costs for all reason types
        from cost_function import cost_function
        from redis_helpers.icf import bitmap_to_icf
        import pandas as pd
        from etl.loader import get_etl_cache
        from etl.progress import ICFProgressMonitor, CacheWriteCoordinator
        
        logger.info("Calculating costs for anti-reasons...")
        
        cache = get_etl_cache()
        dataset_name = db.get('_dataset_name', 'unknown')
        
        # Find the ZIP path for this dataset
        zip_paths = sorted(Path("results").glob(f"{dataset_name}_*.zip"))
        selected_zip_path = zip_paths[0] if zip_paths else None
        
        # Define all reason types to process
        reason_types = ['reasons', 'non_reasons', 'anti_reasons']
        
        # Try to load from cache
        cached_data = None
        if selected_zip_path:
            cached_data = cache.load_costs(selected_zip_path, reason_types)
        
        if cached_data is not None:
            logger.info(f"Loaded costs from cache for {dataset_name}")
            cost_df = cached_data['cost_df']
            # Update tests_sample with cached data
            for sample_id in test_ids:
                if sample_id in cached_data['tests_sample']:
                    for reason_type in reason_types:
                        if reason_type in cached_data['tests_sample'][sample_id]:
                            tests_sample[sample_id][reason_type] = cached_data['tests_sample'][sample_id][reason_type]
        else:
            logger.info(f"Computing costs for {dataset_name} (not in cache)...")
            
            # Import parallel cost calculation
            from etl.parallel_costs import calculate_costs_parallel_incremental
            
            progress_monitor = ICFProgressMonitor(
                project_name=f"{dataset_name} ICF",
                refresh_interval=1.0,
                enabled=verbose
            )
            
            with CacheWriteCoordinator(cache=cache, zip_path=selected_zip_path, verbose=verbose) as cache_writer:
                # Calculate costs in parallel with incremental cache saves
                total_costs, tests_sample = calculate_costs_parallel_incremental(
                    db=db,
                    test_ids=test_ids,
                    tests_sample=tests_sample,
                    sigmas_all=sigmas_all,
                    cost_function=cost_function,
                    bitmap_to_icf=bitmap_to_icf,
                    reason_types=reason_types,
                    cache=cache,
                    selected_zip_path=selected_zip_path,
                    n_workers=None,  # Auto-detect CPU count
                    batch_size=50,    # Process 50 ICFs per batch
                    save_every_n_batches=10,  # Save to cache every 10 batches
                    verbose=verbose,
                    progress_monitor=progress_monitor,
                    cache_writer=cache_writer
                )
            
            logger.info(f"Parallel computation complete: {total_costs} costs calculated")
            
            # Load cost_df from cache (already saved incrementally)
            cached_costs = cache.load_costs(selected_zip_path, reason_types)
            if cached_costs:
                cost_df = cached_costs['cost_df']
            else:
                # Fallback: create empty DataFrame
                cost_df = pd.DataFrame()
        
        # Step 3: Calculate robustness
        num_features = len(feature_names)
        
        logger.info("Calculating robustness metrics...")
        
        if len(cost_df) == 0 or 'reason_type' not in cost_df.columns:
            logger.warning("Cannot calculate robustness - no cost data available")
            return
        
        # Filter only anti_reasons for robustness calculation
        anti_reasons_df = cost_df[cost_df['reason_type'] == 'anti_reasons'].copy()
        
        if len(anti_reasons_df) == 0:
            logger.warning("No anti_reasons found in cost data")
            return
        
        # Calculate robustness for each anti-reason ICF
        from etl.reasons_analysis import calculate_robustness_per_bitmap
        bitmap_robustness = calculate_robustness_per_bitmap(anti_reasons_df, num_features=num_features)
        
        logger.info(f"Calculated robustness for {len(bitmap_robustness)} anti-reason ICFs")
        
        # Calculate robustness for all samples
        from etl.reasons_analysis import calculate_all_samples_robustness, print_robustness_statistics
        
        sample_robustness_df = calculate_all_samples_robustness(
            cost_df=cost_df,
            num_features=num_features,
            tests_sample=tests_sample,
            test_ids=test_ids,
            verbose=verbose
        )
        
        if verbose:
            print_robustness_statistics(sample_robustness_df)
        
        # Step 4: Save results
        sample_robustness_path = results_dir / 'sample_robustness.csv'
        sample_robustness_df.to_csv(sample_robustness_path, index=False)
        logger.info(f"Sample robustness results saved to: {sample_robustness_path}")
        
        anti_reasons_robustness_path = results_dir / 'anti_reasons_robustness.csv'
        bitmap_robustness.to_csv(anti_reasons_robustness_path, index=False)
        logger.info(f"Anti-reasons robustness results saved to: {anti_reasons_robustness_path}")
        
        # Step 5: Generate accuracy vs robustness report
        from etl.tables import build_accuracy_vs_robustness_report
        
        report = build_accuracy_vs_robustness_report(db, dataset_name, sample_robustness_df)
        
        # Save report
        report_path = results_dir / f'{dataset_name}_robustness_report.txt'
        with open(report_path, 'w') as f:
            for line in report['lines']:
                f.write(line + '\n')
                if verbose:
                    print(line)
        
        logger.info(f"Robustness report saved to: {report_path}")
        
        # Step 6: Create visualizations (if requested)
        if verbose:
            try:
                # Create dedicated figures directory
                figures_dir = output_dir / "figures"
                figures_dir.mkdir(exist_ok=True)
                
                from etl.reasons_analysis import create_robustness_visualizations
                
                logger.info("Creating robustness visualizations...")
                fig_main, fig_quartiles = create_robustness_visualizations(
                    sample_robustness_df=sample_robustness_df,
                    dataset_name=dataset_name
                )
                
                # Save visualizations as HTML files in figures directory
                if fig_main is not None:
                    fig_path = figures_dir / f'{dataset_name}_robustness_main.html'
                    fig_main.write_html(str(fig_path))
                    logger.info(f"Main robustness visualization saved to: {fig_path}")
                
                if fig_quartiles is not None:
                    fig_path = figures_dir / f'{dataset_name}_robustness_quartiles.html'
                    fig_quartiles.write_html(str(fig_path))
                    logger.info(f"Quartiles robustness visualization saved to: {fig_path}")
                    
                # Call the comprehensive visualization function
                visualization(
                    anti_reasons_df=anti_reasons_df,
                    num_features=num_features,
                    tests_sample=tests_sample,
                    sample_robustness_df=sample_robustness_df,
                    test_ids=test_ids,
                    feature_names=feature_names,
                    dataset_name=dataset_name,
                    output_dir=output_dir
                )
                    
            except ImportError as e:
                logger.warning(f"Could not create visualizations: {e}")
        
        logger.info("Robustness analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Robustness analysis failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise
    return feature_names, sample_robustness_df, test_ids, cost_df

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze machine learning model results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--force-refresh", 
        action="store_true", 
        help="Force refresh of cached data"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result zip files (default: results)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for CSV files (default: current directory)"
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip robustness analysis (faster execution)"
    )
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("Starting models analysis...")
        
        # Check if results directory exists
        if not args.results_dir.exists():
            logger.error(f"Results directory not found: {args.results_dir}")
            sys.exit(1)
        
        # Ensure output directory exists
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Set up ETL pipeline
        logger.info("Setting up ETL pipeline...")
        db = setup_etl_pipeline(args.results_dir, args.force_refresh, args.verbose)
        
        if db is None:
            logger.error("Failed to load data through ETL pipeline")
            sys.exit(1)
        
 
        
        # Step 5: Perform robustness analysis (from reasons_analysis.ipynb)
        if not args.skip_robustness:
            logger.info("Performing robustness analysis...")
            feature_names, sample_robustness_df, test_ids, cost_df = perform_robustness_analysis(db, args.output_dir, args.verbose)
            
            logger.info("Preparing analysis...")
            analysis_context = prepare_analysis(db, args.verbose)
            from etl.reasons_analysis import calculate_robustness_per_bitmap, calculate_sample_robustness

            num_features, cost_df = len(feature_names)


            # Check if anti_reasons exist in cost_df
            if len(cost_df) == 0 or 'reason_type' not in cost_df.columns:
                print("  Skipping robustness calculation section.")
            else:
                # Filter only anti_reasons for robustness calculation
                anti_reasons_df = cost_df[cost_df['reason_type'] == 'anti_reasons'].copy()

                if len(anti_reasons_df) == 0:
                    print("\n WARNING: No anti_reasons found in cost data.")
                else:
                    # Calculate robustness for each anti-reason ICF
                    # This finds the maximum cost across all samples for each anti-reason
                    bitmap_robustness = calculate_robustness_per_bitmap(anti_reasons_df, num_features=num_features)

        logger.info("Models analysis completed successfully!")
        logger.info(f"Output files saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()