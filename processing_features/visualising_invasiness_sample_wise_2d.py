import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIG ===
csv_path = "/home/ibab/Desktop/Cell_invasiveness/segmentation_results/full_segmentation_features.csv"
output_dir = "invasion_trends_clean"
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(csv_path)

# Ensure correct column types
df['timepoint'] = df['timepoint'].astype(int)
samples = df['subfolder'].unique()

print(f"Found samples: {samples}")

# === PROCESS EACH SAMPLE ===
for sample in samples:
    sample_df = df[df['subfolder'] == sample]
    timepoints = sorted(sample_df['timepoint'].unique())

    # Compute invasion center using t=0 (or first timepoint)
    t0 = sample_df[sample_df['timepoint'] == timepoints[0]]
    cx, cy, cz = t0[['centroid-0', 'centroid-1', 'centroid-2']].mean()

    stats = []
    for t in timepoints:
        tdf = sample_df[sample_df['timepoint'] == t].copy()
        # Euclidean distance from invasion center
        tdf['radial_dist'] = np.sqrt((tdf['centroid-0'] - cx) ** 2 +
                                     (tdf['centroid-1'] - cy) ** 2 +
                                     (tdf['centroid-2'] - cz) ** 2)
        mean_r = tdf['radial_dist'].mean()
        sd_r = tdf['radial_dist'].std()
        n_cells = len(tdf)
        stats.append((t, mean_r, sd_r, n_cells))

    stats_df = pd.DataFrame(stats, columns=['timepoint', 'mean_radius', 'sd_radius', 'cell_count'])

    # === PLOT ===
    plt.figure(figsize=(7, 5))
    plt.errorbar(stats_df['timepoint'], stats_df['mean_radius'],
                 yerr=stats_df['sd_radius'], fmt='-o', capsize=4, label='Mean ± SD radius')
    plt.title(f"Invasion progression over time — {sample}")
    plt.xlabel("Timepoint")
    plt.ylabel("Mean invasion distance (pixels)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    outpath = os.path.join(output_dir, f"{sample}_invasion_trend.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {outpath}")

print("\nAll sample invasion plots saved in:", output_dir)
