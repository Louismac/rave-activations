
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import linregress


print("="*80)
print("HIERARCHICAL PATTERN LINEAR REGRESSION ANALYSIS")
print("="*80)

# Load data
base_path = Path('/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/results/6_cluster')
models = ["strings", "drum_loops", "taylor_vocal"]

# Categorize datasets
natural_datasets = ["strings", "vocals", "drum_loops"]  # Natural audio
synthetic_datasets = ["stimuli"]  # Synthetic stimuli

def extract_layer_depth(layer_name):
    """Extract numeric depth from layer name"""
    import re
    match = re.search(r'net\.(\d+)', layer_name)
    if match:
        return int(match.group(1))
    return None

# Storage for layer-level data
all_data = {
    'natural': {'pitch': [], 'bpm': []},
    'synthetic': {'pitch': [], 'bpm': []}
}

print("\nLoading ALL layer-level data (no binning)...")
print("-"*80)

# Collect all layer-level data
for model in models:
    for dataset in natural_datasets + synthetic_datasets:
        csv_path = base_path / model / dataset / "table1_layer_statistics.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            # Add depth column
            df['depth'] = df['Layer'].apply(extract_layer_depth)
            df = df.dropna(subset=['depth'])
            
            # Store each layer individually (no binning!)
            for _, row in df.iterrows():
                depth = row['depth']
                pitch_corr = abs(row['Pitch Hz Corr'])
                bpm_corr = abs(row['BPM Corr'])
                
                if dataset in natural_datasets:
                    # drum_loops has no pitch info; vocals BPM at chance (permutation baseline)
                    if dataset != 'drum_loops' and not pd.isna(pitch_corr):
                        all_data['natural']['pitch'].append({
                            'model': model,
                            'dataset': dataset,
                            'depth': depth,
                            'correlation': pitch_corr
                        })
                    if dataset != 'vocals':
                        all_data['natural']['bpm'].append({
                            'model': model,
                            'dataset': dataset,
                            'depth': depth,
                            'correlation': bpm_corr
                        })
                else:  # synthetic
                    if not pd.isna(pitch_corr):
                        all_data['synthetic']['pitch'].append({
                            'model': model,
                            'dataset': dataset,
                            'depth': depth,
                            'correlation': pitch_corr
                        })
                    all_data['synthetic']['bpm'].append({
                        'model': model,
                        'dataset': dataset,
                        'depth': depth,
                        'correlation': bpm_corr
                    })

print(f"Natural pitch: {len(all_data['natural']['pitch'])} layer observations")
print(f"Natural BPM: {len(all_data['natural']['bpm'])} layer observations")
print(f"Synthetic pitch: {len(all_data['synthetic']['pitch'])} layer observations")
print(f"Synthetic BPM: {len(all_data['synthetic']['bpm'])} layer observations")

# ============================================================================
# LINEAR REGRESSION ANALYSIS - LAYER LEVEL
# ============================================================================

print("\n" + "="*80)
print("LINEAR REGRESSION RESULTS (LAYER-LEVEL)")
print("="*80)

results = {}

for stimulus_type in ['natural', 'synthetic']:
    print(f"\n{'='*80}")
    print(f"{stimulus_type.upper()} AUDIO")
    print(f"{'='*80}")
    
    for feature in ['pitch', 'bpm']:
        if len(all_data[stimulus_type][feature]) == 0:
            continue
            
        print(f"\n--- {feature.upper()} ---")
        
        data_df = pd.DataFrame(all_data[stimulus_type][feature])
        
        # Show depth range
        print(f"Depth range: {data_df['depth'].min():.0f} to {data_df['depth'].max():.0f}")
        print(f"N observations: {len(data_df)}")
        
        # Compute mean by depth for visualization
        depth_summary = data_df.groupby('depth')['correlation'].agg(['mean', 'std', 'count'])
        
        # Show early/middle/late means for comparison
        early_mean = depth_summary.loc[depth_summary.index <= 7, 'mean'].mean()
        middle_mean = depth_summary.loc[(depth_summary.index > 7) & (depth_summary.index <= 14), 'mean'].mean()
        late_mean = depth_summary.loc[depth_summary.index > 14, 'mean'].mean()
        
        print(f"\nDepth region means (for reference):")
        print(f"  Early (0-7):   {early_mean:.3f}")
        print(f"  Middle (8-14): {middle_mean:.3f}")
        print(f"  Late (15+):    {late_mean:.3f}")
        
        # LINEAR REGRESSION on ALL individual layers
        # x = data_df['depth'].values
        # y = data_df['correlation'].values

        # LINEAR REGRESSION on ALL individual layers aggregates (so not multiple depth measuremnts)
        x = depth_summary.index.values
        y = depth_summary['mean'].values
        print(x,y)

        from sklearn.metrics import r2_score
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        # Calculate predicted values
        y_pred = slope * x + intercept
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        coeffs = np.polyfit(x, y, deg=2)  # Returns [a, b, c] for ax² + bx + c
        poly = np.poly1d(coeffs)
        y_pred_quad = poly(x)
        r2_quad = r2_score(y, y_pred_quad)

        print(f"Linear R²: {r_squared:.3f}")
        print(f"Quadratic R²: {r2_quad:.3f}")
        print(f"Improvement: {r2_quad - r_squared:.3f}")
        
        
        
        print(f"\n--- LINEAR REGRESSION (ALL LAYERS) ---")
        print(f"  Slope (β):          {slope:+.4f} per layer")
        print(f"  Intercept:          {intercept:.4f}")
        print(f"  R-squared:          {r_squared:.4f}")
        print(f"  P-value:            {p_value:.6f}")
        print(f"  Standard error:     {std_err:.5f}")
        
        early_middle_layers = 8
        middle_late_layers = 7
        avg_bin_width = (early_middle_layers + middle_late_layers) / 2
        slope_per_bin = slope * avg_bin_width
        
        print(f"\n--- COMPARISON METRICS ---")
        print(f"  Slope per ~7.5 layers: {slope_per_bin:+.4f} (comparable to bin-based β)")
        print(f"  Early → Late change:   {early_mean:.3f} → {late_mean:.3f}")
        print(f"  Absolute change:       {late_mean - early_mean:+.3f}")
        print(f"  Percent change:        {((late_mean - early_mean) / early_mean) * 100:+.1f}%")
        
        # Significance
        if p_value < 0.001:
            sig_str = "*** HIGHLY SIGNIFICANT (p<0.001)"
        elif p_value < 0.01:
            sig_str = "** SIGNIFICANT (p<0.01)"
        elif p_value < 0.05:
            sig_str = "* SIGNIFICANT (p<0.05)"
        else:
            sig_str = "NOT SIGNIFICANT (p≥0.05)"
        print(f"  Significance:          {sig_str}")
        
        # Store results
        key = f"{stimulus_type}_{feature}"
        results[key] = {
            'slope_per_layer': slope,
            'slope_per_bin': slope_per_bin,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': std_err,
            'n_obs': len(x),
            'early_mean': early_mean,
            'middle_mean': middle_mean,
            'late_mean': late_mean,
            'percent_change': ((late_mean - early_mean) / early_mean) * 100,
            'x': x,
            'y': y,
            'y_pred': y_pred,
            'depth_summary': depth_summary
        }


# ============================================================================
# PLOT LINEAR REGRESSIONS
# ============================================================================

import matplotlib.pyplot as plt

print("\n" + "="*80)
print("PLOTTING LINEAR REGRESSIONS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Hierarchical Organization: Layer Depth vs Feature Correlation', fontsize=14, fontweight='bold')

feature_styles = {
    'pitch': {'color': '#4878CF', 'marker': 'o', 'label': 'Pitch'},
    'bpm':   {'color': '#E84646', 'marker': 's', 'label': 'BPM'},
}

panel_configs = [
    ('natural',   'Natural Audio',   axes[0]),
    ('synthetic', 'Synthetic Audio', axes[1]),
]

for stimulus_type, title, ax in panel_configs:
    for feature, style in feature_styles.items():
        key = f'{stimulus_type}_{feature}'
        if key not in results:
            continue
        res = results[key]
        x, y, y_pred = res['x'], res['y'], res['y_pred']

        sig = ('***' if res['p_value'] < 0.001 else
               '**'  if res['p_value'] < 0.01  else
               '*'   if res['p_value'] < 0.05  else 'n.s.')

        ax.scatter(x, y, alpha=0.6, s=40, color=style['color'],
                   marker=style['marker'], zorder=3)
        ax.plot(x, y_pred, '-', color=style['color'], linewidth=2,
                label=f"{style['label']}  β={res['slope_per_layer']:+.4f}{sig}  R²={res['r_squared']:.3f}")

    ax.set_xlabel('Layer Depth', fontsize=11)
    ax.set_ylabel('|Correlation|', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
base_path = Path("/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/plots") 
output_path = base_path / 'hierarchical_linear_regressions_2panel.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to {output_path}")

plt.close()