"""
Enhanced Invasion Analysis for MDA-MB-231 Cells
Run this script in the same folder as your CSV files:
- full_segmentation_features-1.csv
- test_segmentation_features-1.csv
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: READ AND PREPARE DATA
# ============================================================================
print("="*80)
print("ENHANCED INVASION ANALYSIS")
print("="*80)

print("\n[1] Reading CSV files...")
train_df = pd.read_csv('../segmentation_results/full_segmentation_features.csv')
test_df = pd.read_csv('../segmentation_results/test_segmentation_features.csv')

print(f"    Training data: {len(train_df)} cells")
print(f"    Test data: {len(test_df)} cells")

# Add dataset labels
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'

# Rename subfolder to sample
train_df = train_df.rename(columns={'subfolder': 'sample'})
test_df = test_df.rename(columns={'subfolder': 'sample'})

# Combine
combined_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"    Combined: {len(combined_df)} cells")

# ============================================================================
# STEP 2: CALCULATE ENHANCED METRICS
# ============================================================================

def calculate_enhanced_metrics(group):
    """Calculate 26 invasion and morphology metrics."""

    # Get 3D centroids
    centroids = group[['centroid-0', 'centroid-1', 'centroid-2']].values
    mean_centroid = centroids.mean(axis=0)

    # Calculate distances from mean centroid
    distances = np.linalg.norm(centroids - mean_centroid, axis=1)

    # Basic invasion metrics
    mean_radius = np.mean(distances)
    max_radius = np.max(distances)
    min_radius = np.min(distances)
    sd_radius = np.std(distances)
    median_radius = np.median(distances)

    # Percentiles
    p25_radius = np.percentile(distances, 25)
    p75_radius = np.percentile(distances, 75)
    p90_radius = np.percentile(distances, 90)
    p95_radius = np.percentile(distances, 95)

    # Leader cells (top 10%)
    top_10_threshold = np.percentile(distances, 90)
    leader_mask = distances >= top_10_threshold
    n_leader_cells = np.sum(leader_mask)
    mean_leader_radius = np.mean(distances[leader_mask]) if n_leader_cells > 0 else 0

    # Distribution shape
    radius_skewness = skew(distances)
    radius_kurtosis = kurtosis(distances)

    # Spatial clustering
    if len(centroids) > 1:
        pairwise_dist = pdist(centroids, metric='euclidean')
        dist_matrix = squareform(pairwise_dist)
        np.fill_diagonal(dist_matrix, np.inf)
        nn_dists = np.min(dist_matrix, axis=1)

        mean_nn_dist = np.mean(nn_dists)
        median_nn_dist = np.median(nn_dists)
        sd_nn_dist = np.std(nn_dists)
        dispersion_index = max_radius / mean_radius if mean_radius > 0 else 0
    else:
        mean_nn_dist = median_nn_dist = sd_nn_dist = dispersion_index = 0

    # Morphology
    mean_volume = group['volume'].mean()
    sd_volume = group['volume'].std()
    mean_compactness = group['compactness'].mean()
    sd_compactness = group['compactness'].std()
    mean_extent = group['extent'].mean()
    sd_extent = group['extent'].std()

    # Cell count and density
    cell_count = len(group)
    invasion_volume = (4/3) * np.pi * (max_radius ** 3) if max_radius > 0 else 0
    cell_density = cell_count / invasion_volume if invasion_volume > 0 else 0

    return pd.Series({
        'mean_radius': mean_radius,
        'median_radius': median_radius,
        'max_radius': max_radius,
        'min_radius': min_radius,
        'sd_radius': sd_radius,
        'p25_radius': p25_radius,
        'p75_radius': p75_radius,
        'p90_radius': p90_radius,
        'p95_radius': p95_radius,
        'n_leader_cells': n_leader_cells,
        'mean_leader_radius': mean_leader_radius,
        'radius_skewness': radius_skewness,
        'radius_kurtosis': radius_kurtosis,
        'dispersion_index': dispersion_index,
        'mean_nn_dist': mean_nn_dist,
        'median_nn_dist': median_nn_dist,
        'sd_nn_dist': sd_nn_dist,
        'mean_volume': mean_volume,
        'sd_volume': sd_volume,
        'mean_compactness': mean_compactness,
        'sd_compactness': sd_compactness,
        'mean_extent': mean_extent,
        'sd_extent': sd_extent,
        'cell_count': cell_count,
        'invasion_volume': invasion_volume,
        'cell_density': cell_density
    })

print("\n[2] Calculating enhanced metrics...")
metrics = combined_df.groupby(['dataset', 'sample', 'timepoint']).apply(
    calculate_enhanced_metrics, include_groups=False
).reset_index()

print(f"    ✓ Calculated 26 metrics for {len(metrics)} groups")

# ============================================================================
# STEP 3: SAVE RESULTS
# ============================================================================

print("\n[3] Saving results...")
metrics.to_csv('enhanced_invasion_metrics.csv', index=False)
print("    ✓ Saved: enhanced_invasion_metrics.csv")

# ============================================================================
# STEP 4: GENERATE VISUALIZATIONS
# ============================================================================

print("\n[4] Generating plots...")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Enhanced Invasion Analysis', fontsize=18, fontweight='bold')

colors = {'train': {1: '#1f77b4', 2: '#ff7f0e'},
          'test': {1: '#2ca02c', 2: '#d62728'}}
markers = {'train': 'o', 'test': 's'}

# Define plots
plots = [
    (0, 0, 'mean_radius', 'Mean Radius (μm)', 'Mean Invasion Radius'),
    (0, 1, 'mean_leader_radius', 'Leader Radius (μm)', 'Leader Cell Invasion'),
    (0, 2, 'dispersion_index', 'Dispersion Index', 'Cell Dispersion'),
    (1, 0, 'mean_nn_dist', 'Distance (μm)', 'Nearest Neighbor Distance'),
    (1, 1, 'cell_density', 'Density (×1e6)', 'Cell Density', 1e6),
    (1, 2, 'n_leader_cells', 'Count', 'Leader Cell Count'),
    (2, 0, 'radius_skewness', 'Skewness', 'Distribution Skewness'),
    (2, 1, 'p90_radius', 'P90 Radius (μm)', '90th Percentile Radius'),
    (2, 2, 'cell_count', 'Cell Count', 'Total Cells')
]

for row, col, metric, ylabel, title, *scale_list in plots:
    scale = scale_list[0] if scale_list else 1
    ax = axes[row, col]

    for dataset in ['train', 'test']:
        for sample in [1, 2]:
            data = metrics[(metrics['dataset'] == dataset) &
                          (metrics['sample'] == sample)]
            ax.plot(data['timepoint'], data[metric] * scale,
                   marker=markers[dataset],
                   color=colors[dataset][sample],
                   label=f'{dataset.capitalize()} S{sample}',
                   linewidth=2, markersize=6)

    ax.set_xlabel('Timepoint', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if metric == 'radius_skewness':
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('enhanced_invasion_plots.png', dpi=300, bbox_inches='tight')
print("    ✓ Saved: enhanced_invasion_plots.png")
plt.close()

# ============================================================================
# STEP 5: PRINT SUMMARY
# ============================================================================

print("\n[5] Summary of Key Metrics:")
print("-"*80)

for dataset in ['test', 'train']:
    for sample in [1, 2]:
        data = metrics[(metrics['dataset'] == dataset) &
                      (metrics['sample'] == sample)]

        print(f"\n{dataset.upper()} - Sample {sample}:")
        print(f"  Mean invasion radius: {data['mean_radius'].mean():.2f} μm")
        print(f"  Leader cells (avg): {data['n_leader_cells'].mean():.1f}")
        print(f"  Leader invasion depth: {data['mean_leader_radius'].mean():.2f} μm")
        print(f"  Cell spacing (NN dist): {data['mean_nn_dist'].mean():.2f} μm")
        print(f"  Dispersion index: {data['dispersion_index'].mean():.3f}")
        print(f"  Cell density: {data['cell_density'].mean():.2e} cells/μm³")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nOutput files created:")
print("  1. enhanced_invasion_metrics.csv")
print("  2. enhanced_invasion_plots.png")
print("="*80)
