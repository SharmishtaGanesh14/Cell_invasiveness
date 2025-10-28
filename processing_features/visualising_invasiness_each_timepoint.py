"""
3D Invasion Visualization Across All Timepoints and Samples
-----------------------------------------------------------
This script reads your segmentation CSVs and generates 3D scatter plots
showing the spatial distribution of cells (centroids) for each sample and timepoint.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================================
# 1️⃣ Load data
# ============================================================================
print("="*80)
print("3D INVASION SNAPSHOT VISUALIZATION")
print("="*80)

# Adjust paths if needed
train_df = pd.read_csv('../segmentation_results/full_segmentation_features.csv')
test_df = pd.read_csv('../segmentation_results/test_segmentation_features.csv')

# Label datasets and combine
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'
train_df = train_df.rename(columns={'subfolder': 'sample'})
test_df = test_df.rename(columns={'subfolder': 'sample'})
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Create output folder
os.makedirs("3D_invasion_plots", exist_ok=True)

# ============================================================================
# 2️⃣ Generate 3D plots
# ============================================================================
print("\n[1] Generating 3D scatter plots for each (sample, timepoint)...")

samples = sorted(combined_df['sample'].unique())
timepoints = sorted(combined_df['timepoint'].unique())

for dataset in ['train', 'test']:
    for sample in samples:
        for t in timepoints:
            subset = combined_df[
                (combined_df['dataset'] == dataset) &
                (combined_df['sample'] == sample) &
                (combined_df['timepoint'] == t)
            ]
            if subset.empty:
                continue

            centroids = subset[['centroid-0', 'centroid-1', 'centroid-2']].values
            mean_centroid = centroids.mean(axis=0)

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       s=10, alpha=0.6, label='Cells')
            ax.scatter(*mean_centroid, c='red', s=60, label='Mean Center', edgecolor='k')

            ax.set_title(f"{dataset.upper()} - Sample {sample}, Timepoint {t}", fontsize=12, fontweight='bold')
            ax.set_xlabel('X (μm)')
            ax.set_ylabel('Y (μm)')
            ax.set_zlabel('Z (μm)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f"3D_invasion_plots/{dataset}_sample{sample}_t{t}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"    ✓ Saved: {filename}")

print("\nAll 3D snapshots saved in '3D_invasion_plots/' folder.")
print("="*80)
