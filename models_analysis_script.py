from pathlib import Path
from cost_function import cal_sigmas, cost_function
from etl.tables import (
    prepare_models_analysis,
    build_accuracy_vs_robustness_report
)
from redis_helpers.icf import bitmap_to_icf
import pandas as pd
from etl.loader import get_etl_cache, etl
from etl.progress import ICFProgressMonitor, CacheWriteCoordinator
# Use ETL function to calculate robustness for all samples
from etl.reasons_analysis import *
# Import parallel cost calculation
from etl.parallel_costs import calculate_costs_parallel_incremental

def model_analysis(db, verbose=True, save_plots=False):
    # analysis_context = prepare_models_analysis(db=db, verbose=verbose, selected_dataset=None)
    tests_sample, X_test, test_ids, feature_names = extract_test_samples(db)

    X_train = db["data"]["TRAINING_SET"]["value_json"]["X_train"]
    sigmas_all = cal_sigmas(X_train, X_test, feature_names, test_ids=test_ids)
    if verbose:
        print(f"Extracted {len(test_ids)} test samples\nFeatures: {len(feature_names)}")
        print("Calculating costs for anti_reasons...")
        print("=" * 80)
    cache = get_etl_cache()
    dataset_name = db.get('_dataset_name', 'unknown')

    # Find the ZIP path for this dataset
    RESULTS_DIR = Path("results")
    zip_paths = sorted(RESULTS_DIR.glob(f"{dataset_name}_*.zip"))
    selected_zip_path = zip_paths[0] if zip_paths else None

    # Define all reason types to process
    reason_types = ['reasons', 'non_reasons', 'anti_reasons']

    # Try to load from cache
    cached_data = None
    if selected_zip_path:
        cached_data = cache.load_costs(selected_zip_path, reason_types)

    if cached_data is not None:
        if verbose:
            print(f"\n Loaded costs from cache for {dataset_name}")
            print("=" * 80)
        cost_df = cached_data['cost_df']
        # Update tests_sample with cached data
        for sample_id in test_ids:
            if sample_id in cached_data['tests_sample']:
                for reason_type in reason_types:
                    if reason_type in cached_data['tests_sample'][sample_id]:
                        tests_sample[sample_id][reason_type] = cached_data['tests_sample'][sample_id][reason_type]
    else:
        if verbose:
            print(f"\n→ Computing costs for {dataset_name} (not in cache)...")
            print("Using PARALLEL computation with INCREMENTAL cache to reduce RAM usage")

        progress_monitor = ICFProgressMonitor(
            project_name=f"{dataset_name} ICF",
            refresh_interval=1.0,
            enabled=True
        )

        with CacheWriteCoordinator(cache=cache, zip_path=selected_zip_path, verbose=False) as cache_writer:
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
                verbose=True,
                progress_monitor=progress_monitor,
                cache_writer=cache_writer
            )

        if verbose:
            print(f"\n Parallel computation complete: {total_costs} costs calculated")

        # Load cost_df from cache (already saved incrementally)
        cached_costs = cache.load_costs(selected_zip_path, reason_types)
        if cached_costs:
            cost_df = cached_costs['cost_df']
        else:
            # Fallback: create empty DataFrame
            cost_df = pd.DataFrame()

    num_features = len(feature_names)
    if verbose:
        print(f"\n{'='*80}")
        print(f"COST CALCULATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total costs calculated: {len(cost_df)}")
        print(f"\n\n{'='*80}")
        print(f"  ROBUSTNESS CALCULATION (Anti-Reasons Only)")
        print(f"{'='*80}\n")
        print(f" Configuration:")
        print(f"   • Number of features: {num_features}")
        print(f"\n Formula:")
        print(f"   r(C,x) = 1 - max{{ICF ∈ AR{{C,y}}}} cost_x(ICF) / |features|")
        print(f"\n   where AR{{C,y}} = set of Anti-Reasons for class y")

    # Check if anti_reasons exist in cost_df
    if len(cost_df) == 0 or 'reason_type' not in cost_df.columns:
        print("\n ERROR: Cannot calculate robustness - no cost data available.")
        print("  Please ensure the database contains anti_reasons data.")
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
            if verbose:
                print(f"\n\n{'='*80}")
                print(f"  ICF-LEVEL ROBUSTNESS ANALYSIS (Anti-Reasons)")
                print(f"{'='*80}\n")
                print(f" ICF Summary:")
                print(f"   • Total anti-reason ICFs: {len(bitmap_robustness)}")
                print(f"\n Cost Statistics:")
                print(f"   • Max cost:      {bitmap_robustness['max_cost'].max():.6f}")
                print(f"   • Mean max cost: {bitmap_robustness['max_cost'].mean():.6f}")
                print(f"   • Min max cost:  {bitmap_robustness['max_cost'].min():.6f}")
                print(f"\n Robustness Statistics:")
                print(f"   • Max:  {bitmap_robustness['robustness'].max():.6f}")
                print(f"   • Mean: {bitmap_robustness['robustness'].mean():.6f}")
                print(f"   • Min:  {bitmap_robustness['robustness'].min():.6f}")

            sample_robustness_df = calculate_all_samples_robustness(
                cost_df=cost_df,
                num_features=num_features,
                tests_sample=tests_sample,
                test_ids=test_ids,
                verbose=verbose
            )

            # Print comprehensive statistics
            if verbose:
                print_robustness_statistics(sample_robustness_df)

            # Save results
            sample_robustness_df.to_csv(f'results/{dataset_name}_sample_robustness.csv', index=False)
            if verbose:
                print(f"\n\nSample-level robustness results saved to: results/{dataset_name}_sample_robustness.csv")

            

            sample_robustness_ref = locals().get('sample_robustness_df')
            report = build_accuracy_vs_robustness_report(db, dataset_name, sample_robustness_ref)
            
            # Save the report to a JSON file
            import json
            report_file_path = f'results/{dataset_name}_accuracy_vs_robustness_report.json'
            with open(report_file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            if verbose:
                print(f"Report saved to: {report_file_path}")
                for line in report['lines']:
                    print(line)

            if save_plots:
                if 'sample_robustness_df' in locals() and len(sample_robustness_df) > 0:
                    # Create visualizations using ETL function
                    fig_main, fig_quartiles = create_robustness_visualizations(
                        sample_robustness_df=sample_robustness_df,
                        dataset_name=dataset_name
                    )
                
                    if fig_main is not None:
                        # Save as PDF
                        pdf_path = RESULTS_DIR / f"{dataset_name}_sample_reasons.pdf"
                        fig_main.write_image(str(pdf_path), format='pdf', width=1200, height=700, scale=1)

                    if fig_quartiles is not None:
                        # Save as PDF
                        pdf_path = RESULTS_DIR / f"{dataset_name}_sample_reasons_quartiles.pdf"
                        fig_quartiles.write_image(str(pdf_path), format='pdf', width=1200, height=700, scale=1)
            else:
                print("sample_robustness_df not available. Run previous cells first.")


            sample_id = test_ids[0]
            if verbose:
                print(f"Total test samples available: {len(test_ids)}")
                print(f"Number of features per sample: {len(feature_names)}")


                print(f"\n{'='*80}")
                print(f"Detailed analysis will use Sample ID: {sample_id}")
            if save_plots:  
                
                # Visualize all time series (max 50 samples)
                fig = visualize_all_time_series(tests_sample, test_ids, feature_names, max_samples=50)
                # Save as PDF
                pdf_path = RESULTS_DIR / f"{dataset_name}_sample_time_series.pdf"
                fig.write_image(str(pdf_path), format='pdf', width=1200, height=700, scale=1)
                
                fig = visualize_sample_with_icf(sample_id, tests_sample, feature_names, reason_type='reasons')
                if fig is not None:
                    
                    # Save as PDF
                    pdf_path = RESULTS_DIR / f"{dataset_name}_sample_visualize_sample_with_icf_reasons.pdf"
                    fig.write_image(str(pdf_path), format='pdf', width=1200, height=700, scale=1)

                fig = visualize_sample_with_icf(sample_id, tests_sample, feature_names, reason_type='anti_reasons')
                if fig is not None:
                    # Save as PDF
                    pdf_path = RESULTS_DIR / f"{dataset_name}_sample_visualize_sample_with_icf_anti_reasons.pdf"
                    fig.write_image(str(pdf_path), format='pdf', width=1200, height=700, scale=1)
                # Plot 4: Combined view - Reason vs Anti-Reason

                fig = visualize_sample_comparison(sample_id, tests_sample, feature_names)
                if fig is not None:
                    # Save as PDF
                    pdf_path = RESULTS_DIR / f"{dataset_name}_sample_visualize_sample_comparison.pdf"
                    fig.write_image(str(pdf_path), format='pdf', width=1200, height=700, scale=1)

                # Calculate robustness per bitmap (ICF) - ONLY for Anti-Reasons
            
                print("\n" + "=" * 80)
                # Filter only anti-reasons
                print("ROBUSTNESS PER ICF (Anti-Reasons Only)")
                print("=" * 80)
                # Calculate with normalization
            bitmap_robustness = calculate_robustness_per_bitmap(anti_reasons_df, num_features=num_features)
            if verbose:
                print(f"\nTotal anti-reason ICFs analyzed: {len(bitmap_robustness)}")
                print(f"\nTop 10 ICFs by max_cost (hardest to reach):")
                print(bitmap_robustness[['max_cost', 'robustness', 'mean_cost', 'n_samples']].head(10).to_string(index=False))

                # Anti-reasons with lowest robustness (easiest to change classification)
                print(f"\n\nANTI-REASONS with LOWEST robustness (most vulnerable - easiest to perturb):")
                ar_lowest = bitmap_robustness.nsmallest(5, 'robustness')
                print(ar_lowest[['max_cost', 'robustness', 'mean_cost', 'n_samples']].to_string(index=False))

            # Save results (including bitmap_index for reference, but not displayed)
            bitmap_robustness.to_csv(f'results/{dataset_name}_anti_reasons_robustness.csv', index=False)
           
            if verbose:
                print(f"\n\nResults saved to: results/{dataset_name}_anti_reasons_robustness.csv")
            if save_plots:
                fig = visualize_sample_comparison_smooth(sample_id, tests_sample, feature_names)
                if fig is not None:
                    pdf_path = RESULTS_DIR / f"{dataset_name}_sample_visualize_sample_comparison_smooth.pdf"
                    fig.write_image(str(pdf_path), format='pdf', width=1200, height=700, scale=1)


if __name__ == "__main__":
    RESULTS_DIR = Path("results")
    zip_paths = sorted(RESULTS_DIR.glob("*.zip"))
    zip_names = [path.name for path in zip_paths]

    for name in zip_names:
        print(f"Processing dataset ZIP: {name}")
        dataset_name = Path(name).stem.split('_')[0]
        existing_csvs = list(RESULTS_DIR.glob(f"{dataset_name}_*.csv"))
        
        # if existing_csvs:
        #     print(f"Skipping dataset '{dataset_name}' - CSV files already exist:")
        #     continue

        if name == 'HandOutlines_0_false_0.zip':
            print(f"Skipping dataset '{dataset_name}' as raises errors.")
            continue
        db = etl(
            zip_paths,
            RESULTS_DIR,
            use_cache=True,           # Use the unified cache system
            force_refresh=False,      # Set to True to force refresh
            auto_select=False,
            load_only_db10=False,      # Load ONLY DB10
            verbose=False,
            name_dataset=name,    # Set to specific dataset name if needed
        )
    
        model_analysis(db, verbose=False, save_plots=False)
#         IEEE_CAI$ nohup python models_analysis_script.py  > models_robustness_def.log 2>&1 & 
# [1] 2737293