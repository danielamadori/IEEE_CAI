#!/usr/bin/env python3
"""
Update combined_analyzed_results_updated.csv to include standard deviations 
with the robustness means in the format "mean (± std)"
"""

import pandas as pd
import numpy as np

# Load the datasets
print("Loading datasets...")
combined_df = pd.read_csv("combined_analyzed_results_updated.csv", index_col=0)
individual_stats_df = pd.read_csv("results/individual_dataset_robustness_stats.csv")

print(f"Combined dataframe shape: {combined_df.shape}")
print(f"Individual stats shape: {individual_stats_df.shape}")

# Create a mapping of dataset -> robustness std for the "Overall" category
robustness_std_map = {}
overall_stats = individual_stats_df[individual_stats_df['category'] == 'Overall']

for _, row in overall_stats.iterrows():
    dataset = row['dataset']
    std_value = row['std']
    robustness_std_map[dataset] = std_value

print(f"\nFound robustness std data for {len(robustness_std_map)} datasets:")
for dataset, std_val in robustness_std_map.items():
    print(f"  {dataset}: {std_val:.6f}")

# Update the Robustness Mean row to include standard deviations
if 'Robustness Mean' in combined_df.index:
    print("\nUpdating Robustness Mean with standard deviations...")
    
    # Create a new row with mean ± std format
    new_robustness_row = pd.Series(index=combined_df.columns, dtype=object)
    
    for dataset in combined_df.columns:
        mean_value = combined_df.loc['Robustness Mean', dataset]
        
        if pd.notna(mean_value) and dataset in robustness_std_map:
            mean_float = float(mean_value)
            std_float = robustness_std_map[dataset]
            new_robustness_row[dataset] = f"{mean_float:.6f} (± {std_float:.6f})"
        elif pd.notna(mean_value):
            # Keep the original value if no std data available
            new_robustness_row[dataset] = f"{float(mean_value):.6f}"
        else:
            # Keep empty if no mean data
            new_robustness_row[dataset] = ""
    
    # Replace the old row with the new formatted one
    combined_df.loc['Robustness Mean (± Std)'] = new_robustness_row
    
    # Remove the old Robustness Mean row
    combined_df = combined_df.drop('Robustness Mean')
    
    print("✅ Successfully updated Robustness Mean with standard deviations")
    
    # Show some examples of the updated format
    print("\nExample updated values:")
    for i, (dataset, value) in enumerate(new_robustness_row.items()):
        if value and i < 5:  # Show first 5 non-empty values
            print(f"  {dataset}: {value}")
else:
    print("❌ 'Robustness Mean' row not found in combined dataframe")

# Save the updated dataframe
output_file = "combined_analyzed_results_updated.csv"
combined_df.to_csv(output_file)
print(f"\n✅ Saved updated file: {output_file}")

# Display the updated dataframe structure
print(f"\n📊 Updated dataframe shape: {combined_df.shape}")
print(f"📋 Metrics now include:")

# Show which robustness-related metrics are now present
robustness_metrics = [idx for idx in combined_df.index if 'Robustness' in idx]
for metric in robustness_metrics:
    print(f"  • {metric}")

print("\n🎯 Summary:")
print(f"   • Updated {len([col for col in combined_df.columns if pd.notna(combined_df.loc['Robustness Mean (± Std)', col]) and '±' in str(combined_df.loc['Robustness Mean (± Std)', col])])} datasets with mean ± std format")
print(f"   • Robustness means now include standard deviations where available")