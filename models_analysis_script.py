#!/usr/bin/env python3
"""
Models Analysis Script

This script performs comprehensive analysis on machine learning model results.
It loads data from zip files, processes it through an ETL pipeline, and generates
various analysis tables and CSV outputs with model performance metrics.

Usage:
    python models_analysis_script.py [--force-refresh] [--verbose]

Output files:
    - combined_analyzed_results.csv: Main analysis results
    - combined_analyzed_results_updated.csv: Enhanced results with additional metrics
    - summary_results.csv: Summary statistics
    - analyzed_counts.csv: Detailed counts and metrics
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
            load_only_db10=True,  # Load ONLY DB10
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
        
        # Step 2: Prepare analysis
        logger.info("Preparing analysis...")
        analysis_context = prepare_analysis(db, args.verbose)
        
        # Step 3: Save basic results
        logger.info("Saving analysis results...")
        save_analysis_results(analysis_context, args.output_dir)
        
        
        
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