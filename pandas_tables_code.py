# =====================================================================
# CODICE DA COPIARE NEL NOTEBOOK reasons_analysis.ipynb
# Sostituire la cella che contiene "import plotly.graph_objects as go"
# con questo codice per avere tabelle Pandas invece di Plotly
# =====================================================================

sample_robustness_df = pd.DataFrame(sample_robustness_list)

# Separate samples by prediction correctness
correct_samples_df = sample_robustness_df[sample_robustness_df['prediction_correct'] == True].copy()
incorrect_samples_df = sample_robustness_df[sample_robustness_df['prediction_correct'] == False].copy()

print("="*100)
print("SAMPLE ROBUSTNESS ANALYSIS - PANDAS DATAFRAMES")
print("="*100)

# 1. TABELLA TUTTI I SAMPLES (ordinata per robustness)
print("\n TABELLA COMPLETA - TUTTI I SAMPLES")
all_samples_table = sample_robustness_df.copy()
status_map = {True: 'Correct', False: 'Incorrect', None: '?'}
all_samples_table['status'] = all_samples_table['prediction_correct'].map(status_map)
all_samples_table = all_samples_table[['status', 'sample_id', 'robustness', 'predicted_label', 'actual_label']]
all_samples_table = all_samples_table.sort_values('robustness', ascending=False).reset_index(drop=True)
all_samples_table.index += 1  # Start index from 1

# Display with styling
styled_all = all_samples_table.style\
    .format({'robustness': '{:.6f}'})\
    .background_gradient(subset=['robustness'], cmap='RdYlGn', vmin=0.7, vmax=0.8)\
    .set_properties(**{'text-align': 'left'})\
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#4472C4'), ('color', 'white'), ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('border', '1px solid #ddd')]},
        {'selector': 'tr:hover', 'props': [('background-color', '#f5f5f5')]}
    ])

display(styled_all)

# 2. TABELLA SOLO PREDIZIONI CORRETTE
if len(correct_samples_df) > 0:
    print(f"\n PREDIZIONI CORRETTE (n={len(correct_samples_df)})")
    correct_table = correct_samples_df[['sample_id', 'robustness', 'predicted_label', 'actual_label']].copy()
    correct_table = correct_table.sort_values('robustness', ascending=False).reset_index(drop=True)
    correct_table.index += 1

    styled_correct = correct_table.style\
        .format({'robustness': '{:.6f}'})\
        .background_gradient(subset=['robustness'], cmap='Greens', vmin=correct_table['robustness'].min(), vmax=correct_table['robustness'].max())\
        .set_properties(**{'text-align': 'left', 'background-color': '#e8f5e9'})\
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#2e7d32'), ('color', 'white'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('border', '1px solid #ddd')]},
        ])

    display(styled_correct)
else:
    print(f"\n PREDIZIONI CORRETTE (n=0) - Nessun sample trovato")

# 3. TABELLA SOLO PREDIZIONI SBAGLIATE
if len(incorrect_samples_df) > 0:
    print(f"\n PREDIZIONI SBAGLIATE (n={len(incorrect_samples_df)})")
    incorrect_table = incorrect_samples_df[['sample_id', 'robustness', 'predicted_label', 'actual_label']].copy()
    incorrect_table = incorrect_table.sort_values('robustness', ascending=False).reset_index(drop=True)
    incorrect_table.index += 1

    styled_incorrect = incorrect_table.style\
        .format({'robustness': '{:.6f}'})\
        .background_gradient(subset=['robustness'], cmap='Reds', vmin=incorrect_table['robustness'].min(), vmax=incorrect_table['robustness'].max())\
        .set_properties(**{'text-align': 'left', 'background-color': '#ffebee'})\
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#c62828'), ('color', 'white'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('border', '1px solid #ddd')]},
        ])

    display(styled_incorrect)
else:
    print(f"\n PREDIZIONI SBAGLIATE (n=0) - Nessun sample trovato")

# 4. TABELLA STATISTICHE RIASSUNTIVE
print("\n STATISTICHE RIASSUNTIVE")
stats_data = []
for label, df_subset in [("All Samples", sample_robustness_df),
                          (" Correct", correct_samples_df),
                          (" Incorrect", incorrect_samples_df)]:
    if len(df_subset) > 0 and df_subset['robustness'].notna().any():
        stats_data.append({
            'Category': label,
            'Count': len(df_subset),
            'Mean': df_subset['robustness'].mean(),
            'Std': df_subset['robustness'].std(),
            'Min': df_subset['robustness'].min(),
            'Q25': df_subset['robustness'].quantile(0.25),
            'Median': df_subset['robustness'].median(),
            'Q75': df_subset['robustness'].quantile(0.75),
            'Max': df_subset['robustness'].max()
        })

if stats_data:
    stats_df = pd.DataFrame(stats_data)
    styled_stats = stats_df.style\
        .format({
            'Mean': '{:.6f}',
            'Std': '{:.6f}',
            'Min': '{:.6f}',
            'Q25': '{:.6f}',
            'Median': '{:.6f}',
            'Q75': '{:.6f}',
            'Max': '{:.6f}'
        })\
        .background_gradient(subset=['Mean'], cmap='YlOrRd')\
        .set_properties(**{'text-align': 'center'})\
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#37474f'), ('color', 'white'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('border', '1px solid #ddd')]},
        ])

    display(styled_stats)

# 5. BOX PLOT con matplotlib
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Box plot 1: All samples
axes[0].boxplot(sample_robustness_df['robustness'].dropna(), widths=0.6)
axes[0].set_title(f'All Samples\n(n={len(sample_robustness_df)})', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Robustness r(C,x)', fontsize=11)
axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Threshold 0.5')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# Box plot 2: Correctly predicted
if len(correct_samples_df) > 0:
    bp2 = axes[1].boxplot(correct_samples_df['robustness'].dropna(), widths=0.6,
                           patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    axes[1].set_title(f' Correctly Predicted\n(n={len(correct_samples_df)})', fontsize=12, fontweight='bold', color='green')
    axes[1].set_ylabel('Robustness r(C,x)', fontsize=11)
    axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
else:
    axes[1].text(0.5, 0.5, 'No data', ha='center', va='center')
    axes[1].set_title(' Correctly Predicted\n(n=0)', fontsize=12, fontweight='bold')

# Box plot 3: Incorrectly predicted
if len(incorrect_samples_df) > 0:
    bp3 = axes[2].boxplot(incorrect_samples_df['robustness'].dropna(), widths=0.6,
                           patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    axes[2].set_title(f' Incorrectly Predicted\n(n={len(incorrect_samples_df)})', fontsize=12, fontweight='bold', color='red')
    axes[2].set_ylabel('Robustness r(C,x)', fontsize=11)
    axes[2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
else:
    axes[2].text(0.5, 0.5, 'No data', ha='center', va='center')
    axes[2].set_title(' Incorrectly Predicted\n(n=0)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('fig/sample_robustness_boxplots.png', dpi=300, bbox_inches='tight')
print("\n Box plots saved to: fig/sample_robustness_boxplots.png")
plt.show()

# 6. HISTOGRAM comparison
if len(correct_samples_df) > 0 or len(incorrect_samples_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))

    if len(correct_samples_df) > 0:
        ax.hist(correct_samples_df['robustness'], bins=20, alpha=0.7, color='green',
                label=f' Correct (n={len(correct_samples_df)})', edgecolor='black')
        correct_mean = correct_samples_df['robustness'].mean()
        ax.axvline(correct_mean, color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Mean Correct: {correct_mean:.4f}')

    if len(incorrect_samples_df) > 0:
        ax.hist(incorrect_samples_df['robustness'], bins=20, alpha=0.7, color='red',
                label=f' Incorrect (n={len(incorrect_samples_df)})', edgecolor='black')
        incorrect_mean = incorrect_samples_df['robustness'].mean()
        ax.axvline(incorrect_mean, color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean Incorrect: {incorrect_mean:.4f}')

    ax.set_xlabel('Robustness r(C,x)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Robustness Distribution: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig/sample_robustness_histogram.png', dpi=300, bbox_inches='tight')
    print(" Histogram saved to: fig/sample_robustness_histogram.png")
    plt.show()

