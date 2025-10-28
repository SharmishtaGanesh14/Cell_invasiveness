"""
Enhanced Invasion Analysis for MDA-MB-231 Cells (Corrected Version)

Author: Shruthi
Date: October 28, 2025

Key Corrections:
- Fixes invasion center to t=0 spheroid position (prevents floating center artifact)
- Applies voxel scaling (1.242 × 1.242 × 6.0 µm) for physical units
- Adds leader fraction metric (proportion rather than count)
- Produces physically meaningful invasion metrics for 3D collagen invasion

Dataset: Cell Tracking Challenge Fluo-C3DL-MDA231
Imaging: Olympus FluoView F1000, Plan 20×/0.7, ~1.242×1.242×6.0 µm voxels
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: READ AND PREPARE DATA
# =============================================================================

print("="*80)
print("ENHANCED INVASION ANALYSIS (Corrected for Biological Meaning)")
print("="*80)

print("\n[1] Reading CSV files...")
train_df = pd.read_csv('../../segmentation_results/full_segmentation_features.csv')
test_df = pd.read_csv('../../segmentation_results/test_segmentation_features.csv')

print(f"   Training data: {len(train_df)} cells")
print(f"   Test data: {len(test_df)} cells")

# Add dataset labels
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'

# Rename subfolder to sample
train_df = train_df.rename(columns={'subfolder': 'sample'})
test_df = test_df.rename(columns={'subfolder': 'sample'})

# Combine datasets
combined_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"   Combined: {len(combined_df)} cells")

# -----------------------------------------------------------------------------
# KEY FIX #1: Define FIXED spheroid center from first timepoint (t=0) per sample
# This prevents the "floating center" artifact that masks real invasion
# -----------------------------------------------------------------------------

print("\n[1b] Computing fixed invasion centers from t=0...")
centers = combined_df.groupby(['dataset', 'sample']).apply(
    lambda g: g[g['timepoint'] == g['timepoint'].min()][['centroid-0', 'centroid-1', 'centroid-2']].mean()
)
centers.columns = ['center_x', 'center_y', 'center_z']

# Merge center info back into combined_df
combined_df = combined_df.merge(centers, on=['dataset', 'sample'], how='left')
print(f"   ✓ Fixed centers computed for {len(centers)} samples")

# =============================================================================
# STEP 2: CALCULATE ENHANCED METRICS (with fixed center + voxel scaling)
# =============================================================================

def calculate_enhanced_metrics(group):
    """
    Calculate 3D invasion and morphology metrics using fixed spheroid center.

    Key Features:
    - Uses fixed t=0 center (not floating mean)
    - Applies voxel scaling for physical units (µm)
    - Computes leader fraction (proportion, not count)
    - Returns biologically interpretable invasion metrics
    """

    # KEY FIX #2: Apply voxel scaling to convert pixel → µm
    # Z-axis is ~5× larger per voxel than XY, so this is critical
    voxel = np.array([1.242, 1.242, 6.0])

    # Extract centroids and fixed center
    centroids = group[['centroid-0', 'centroid-1', 'centroid-2']].values
    center = group[['center_x', 'center_y', 'center_z']].iloc[0].values

    # Compute distances (in µm) from fixed spheroid center
    scaled_diff = (centroids - center) * voxel
    distances = np.linalg.norm(scaled_diff, axis=1)

    # Basic invasion metrics
    mean_radius = np.mean(distances)
    median_radius = np.median(distances)
    max_radius = np.max(distances)
    min_radius = np.min(distances)
    sd_radius = np.std(distances)

    # Percentiles (useful for tracking leading edge)
    p25_radius = np.percentile(distances, 25)
    p75_radius = np.percentile(distances, 75)
    p90_radius = np.percentile(distances, 90)
    p95_radius = np.percentile(distances, 95)

    # KEY FIX #3: Leader cells (top 10%) — now includes FRACTION
    top_10_threshold = np.percentile(distances, 90)
    leader_mask = distances >= top_10_threshold
    n_leader_cells = np.sum(leader_mask)
    leader_fraction = n_leader_cells / len(distances) if len(distances) > 0 else 0
    mean_leader_radius = np.mean(distances[leader_mask]) if n_leader_cells > 0 else 0

    # Distribution shape (positive skew = leading edge extends far)
    radius_skewness = skew(distances)
    radius_kurtosis = kurtosis(distances)

    # Spatial clustering metrics (nearest neighbor analysis)
    if len(centroids) > 1:
        pairwise_dist = pdist(centroids * voxel, metric='euclidean')
        dist_matrix = squareform(pairwise_dist)
        np.fill_diagonal(dist_matrix, np.inf)
        nn_dists = np.min(dist_matrix, axis=1)
        mean_nn_dist = np.mean(nn_dists)
        median_nn_dist = np.median(nn_dists)
        sd_nn_dist = np.std(nn_dists)
        dispersion_index = max_radius / mean_radius if mean_radius > 0 else 0
    else:
        mean_nn_dist = median_nn_dist = sd_nn_dist = dispersion_index = 0

    # Morphology metrics
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
        'leader_fraction': leader_fraction,
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

print("\n[2] Calculating enhanced metrics with corrected invasion center and scaling...")
metrics = combined_df.groupby(['dataset', 'sample', 'timepoint']).apply(
    calculate_enhanced_metrics, include_groups=False
).reset_index()
print(f"   ✓ Calculated metrics for {len(metrics)} timepoints across samples")

# =============================================================================
# STEP 3: SAVE RESULTS
# =============================================================================

print("\n[3] Saving results...")
metrics.to_csv('enhanced_invasion_metrics_corrected.csv', index=False)
print("   ✓ Saved: enhanced_invasion_metrics_corrected.csv")

# =============================================================================
# STEP 4: GENERATE VISUALIZATIONS
# =============================================================================

print("\n[4] Generating plots...")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Corrected 3D Invasion Analysis (MDA-MB-231)', fontsize=18, fontweight='bold')

colors = {'train': {1: '#1f77b4', 2: '#ff7f0e'},
          'test': {1: '#2ca02c', 2: '#d62728'}}
markers = {'train': 'o', 'test': 's'}

# KEY FIX #4: Added leader_fraction plot (row 1, col 2)
plots = [
    (0, 0, 'mean_radius', 'Mean Radius (µm)', 'Mean Invasion Radius'),
    (0, 1, 'mean_leader_radius', 'Leader Radius (µm)', 'Leader Cell Invasion Depth'),
    (0, 2, 'dispersion_index', 'Dispersion Index', 'Cell Dispersion'),
    (1, 0, 'mean_nn_dist', 'NN Distance (µm)', 'Nearest Neighbor Spacing'),
    (1, 1, 'cell_density', 'Density (×1e6)', 'Cell Density', 1e6),
    (1, 2, 'leader_fraction', 'Fraction', 'Leader Fraction Over Time'),
    (2, 0, 'radius_skewness', 'Skewness', 'Distribution Skewness'),
    (2, 1, 'p90_radius', 'P90 Radius (µm)', '90th Percentile Radius'),
    (2, 2, 'cell_count', 'Cell Count', 'Total Cells')
]

for row, col, metric, ylabel, title, *scale_list in plots:
    scale = scale_list[0] if scale_list else 1
    ax = axes[row, col]

    for dataset in ['train', 'test']:
        for sample in [1, 2]:
            data = metrics[(metrics['dataset'] == dataset) & (metrics['sample'] == sample)]
            if len(data) == 0:
                continue

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

    # Add reference line for skewness
    if metric == 'radius_skewness':
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('enhanced_invasion_plots_corrected.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: enhanced_invasion_plots_corrected.png")
plt.close()

# =============================================================================
# STEP 5: PRINT SUMMARY
# =============================================================================

print("\n[5] Summary of Key Metrics:")
print("-"*80)

for dataset in ['test', 'train']:
    for sample in [1, 2]:
        data = metrics[(metrics['dataset'] == dataset) & (metrics['sample'] == sample)]
        if len(data) == 0:
            continue

        print(f"\n{dataset.upper()} - Sample {sample}:")
        print(f"   Mean invasion radius: {data['mean_radius'].mean():.2f} µm")
        print(f"   Leader fraction (avg): {data['leader_fraction'].mean():.2f}")
        print(f"   Leader invasion depth: {data['mean_leader_radius'].mean():.2f} µm")
        print(f"   Mean NN distance: {data['mean_nn_dist'].mean():.2f} µm")
        print(f"   Dispersion index: {data['dispersion_index'].mean():.3f}")
        print(f"   Cell density: {data['cell_density'].mean():.2e} cells/µm³")

print("\n" + "="*80)
print("CORRECTED ANALYSIS COMPLETE ✅")
print("="*80)
print("\nOutput files created:")
print("   1. enhanced_invasion_metrics_corrected.csv")
print("   2. enhanced_invasion_plots_corrected.png")
print("="*80)
print("\nKey improvements over original code:")
print("   ✓ Fixed center computed from t=0 (prevents floating center artifact)")
print("   ✓ Voxel scaling applied (1.242×1.242×6.0 µm)")
print("   ✓ Leader fraction metric added")
print("   ✓ All distances now in physical units (µm)")
print("="*80)
