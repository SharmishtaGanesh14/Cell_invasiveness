"""
3D Invasion Progression Visualization
-------------------------------------
Creates one 3D scatter plot per sample showing all timepoints together,
colored by timepoint to visualize invasion expansion over time.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.cm as cm

# ============================================================================
# 1️⃣ Load and prepare data
# ============================================================================
print("="*80)
print("3D INVASION OVER TIME VISUALIZATION")
print("="*80)

# Adjust paths if needed
train_df = pd.read_csv('../segmentation_results/full_segmentation_features.csv')
test_df = pd.read_csv('../segmentation_results/test_segmentation_features.csv')

train_df['dataset'] = 'train'
test_df['dataset'] = 'test'

# Rename for consistency
train_df = train_df.rename(columns={'subfolder': 'sample'})
test_df = test_df.rename(columns={'subfolder': 'sample'})

combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Output directory
os.makedirs("3D_invasion_plots_combined", exist_ok=True)

# ============================================================================
# 2️⃣ Plot one figure per sample showing all timepoints
# ============================================================================
print("\n[1] Plotting one 3D plot per sample showing all timepoints...")

samples = sorted(combined_df['sample'].unique())
datasets = ['train', 'test']

for dataset in datasets:
    for sample in samples:
        subset = combined_df[
            (combined_df['dataset'] == dataset) &
            (combined_df['sample'] == sample)
        ]
        if subset.empty:
            continue

        timepoints = sorted(subset['timepoint'].unique())
        cmap = cm.get_cmap('viridis', len(timepoints))  # color per timepoint

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each timepoint with a unique color
        for i, t in enumerate(timepoints):
            tp_data = subset[subset['timepoint'] == t]
            centroids = tp_data[['centroid-0', 'centroid-1', 'centroid-2']].values
            mean_centroid = centroids.mean(axis=0)
            ax.scatter(
                centroids[:, 0], centroids[:, 1], centroids[:, 2],
                color=cmap(i), s=10, alpha=0.5, label=f'Timepoint {t}'
            )
            # Optionally mark center per timepoint
            ax.scatter(*mean_centroid, c=[cmap(i)], s=40, edgecolor='k', marker='o')

        ax.set_title(f"{dataset.upper()} - Sample {sample}: Invasion Progression", fontsize=13, fontweight='bold')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.legend(loc='upper left', fontsize=8, title="Timepoints")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"3D_invasion_plots_combined/{dataset}_sample{sample}_progression.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved: {filename}")

print("\nAll combined 3D invasion progression plots saved in '3D_invasion_plots_combined/' folder.")
print("="*80)
