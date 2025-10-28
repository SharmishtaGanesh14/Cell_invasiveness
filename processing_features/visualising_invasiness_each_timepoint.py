import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =============================================================================
# Configuration
# =============================================================================
VOXEL_SIZE = np.array([1.242, 1.242, 6.0])  # voxel size in microns (X, Y, Z)
LEADER_PERCENTILE = 90  # top 10% leader cells by invasion distance

print("=" * 80)
print("3D INVASION VISUALIZATION (Simplified and Corrected)")
print("=" * 80)

# =============================================================================
# 1️⃣ Load and combine data
# =============================================================================
train_df = pd.read_csv('../segmentation_results/full_segmentation_features.csv')
test_df = pd.read_csv('../segmentation_results/test_segmentation_features.csv')

train_df = train_df.rename(columns={'subfolder': 'sample'})
test_df = test_df.rename(columns={'subfolder': 'sample'})

train_df['dataset'] = 'train'
test_df['dataset'] = 'test'

combined_df = pd.concat([train_df, test_df], ignore_index=True)
print("Columns in combined_df:", combined_df.columns.tolist())

# =============================================================================
# 2️⃣ Compute fixed spheroid centers per dataset & sample (from timepoint 0)
# =============================================================================
centers = (
    combined_df[combined_df['timepoint'] == combined_df['timepoint'].min()]
    .groupby(['dataset', 'sample'], as_index=False)[['centroid-0', 'centroid-1', 'centroid-2']]
    .mean()
)
centers.columns = ['dataset', 'sample', 'center_x', 'center_y', 'center_z']

combined_df = combined_df.merge(centers, on=['dataset', 'sample'], how='left')

# =============================================================================
# 3️⃣ Compute invasion distance from fixed centers (with voxel scaling)
# =============================================================================
def compute_distance(row):
    centroid = row[['centroid-0', 'centroid-1', 'centroid-2']].values
    center = row[['center_x', 'center_y', 'center_z']].values
    diff = (centroid - center) * VOXEL_SIZE
    return np.linalg.norm(diff)

combined_df['invasion_distance'] = combined_df.apply(compute_distance, axis=1)

# =============================================================================
# 4️⃣ Identify leader cells (top 10% invasion distance per group)
# =============================================================================
def assign_leaders(group):
    thresh = np.percentile(group['invasion_distance'], LEADER_PERCENTILE)
    group['is_leader'] = group['invasion_distance'] >= thresh
    return group

# ✅ fixed version — keep sample/timepoint columns intact
combined_df = (
    combined_df.groupby(['dataset', 'sample', 'timepoint'], group_keys=False)
    .apply(assign_leaders)
    .reset_index(drop=True)
)

print("Columns after assigning leaders:", combined_df.columns.tolist())
print(f"Leader cells identified: {combined_df['is_leader'].sum()} out of {len(combined_df)} total")

# =============================================================================
# 5️⃣ Create output folders
# =============================================================================
os.makedirs('3D_invasion_plots_corrected', exist_ok=True)
os.makedirs('3D_invasion_videos_corrected', exist_ok=True)

# =============================================================================
# 6️⃣ Generate 3D scatter plots
# =============================================================================
samples = sorted(combined_df['sample'].unique())
timepoints = sorted(combined_df['timepoint'].unique())

x_coords = combined_df['centroid-0'].values * VOXEL_SIZE[0]
y_coords = combined_df['centroid-1'].values * VOXEL_SIZE[1]
z_coords = combined_df['centroid-2'].values * VOXEL_SIZE[2]

x_min, x_max = x_coords.min(), x_coords.max()
y_min, y_max = y_coords.min(), y_coords.max()
z_min, z_max = z_coords.min(), z_coords.max()

count = 0

for dataset in ['train', 'test']:
    for sample in samples:
        for tp in timepoints:
            sub = combined_df[
                (combined_df['dataset'] == dataset)
                & (combined_df['sample'] == sample)
                & (combined_df['timepoint'] == tp)
            ]
            if sub.empty:
                continue

            coords = sub[['centroid-0', 'centroid-1', 'centroid-2']].values * VOXEL_SIZE
            center = sub[['center_x', 'center_y', 'center_z']].iloc[0].values * VOXEL_SIZE
            distances = sub['invasion_distance'].values
            leader_mask = sub['is_leader'].values

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(
                coords[~leader_mask, 0], coords[~leader_mask, 1], coords[~leader_mask, 2],
                c=distances[~leader_mask], cmap='viridis', s=30, alpha=0.6,
                label='Follower Cells', edgecolors='none'
            )
            ax.scatter(
                coords[leader_mask, 0], coords[leader_mask, 1], coords[leader_mask, 2],
                c='red', s=80, alpha=0.8,
                label=f'Leader Cells (n={leader_mask.sum()})',
                edgecolors='darkred', linewidths=1
            )
            ax.scatter(
                *center, c='gold', s=200, marker='*', label='Fixed Center (t=0)',
                edgecolor='black', linewidths=2
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            ax.set_title(
                f"{dataset.upper()} - Sample {sample}, Timepoint {tp}\n"
                f"Leader Fraction: {leader_mask.sum() / len(leader_mask) * 100:.1f}% | "
                f"Max Distance: {distances.max():.1f} µm",
                fontsize=12, fontweight='bold'
            )
            ax.set_xlabel('X (µm)', fontweight='bold')
            ax.set_ylabel('Y (µm)', fontweight='bold')
            ax.set_zlabel('Z (µm)', fontweight='bold')

            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label('Invasion Distance (µm)', fontweight='bold')

            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

            filename = f'3D_invasion_plots_corrected/{dataset}_sample{sample}_t{tp:02d}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            count += 1
            if count % 5 == 0:
                print(f"{count} plots generated...")

print(f"\nTotal plots saved: {count}")
print("Static 3D snapshots saved to '3D_invasion_plots_corrected/' folder.")

# =============================================================================
# 7️⃣ Create time-lapse videos
# =============================================================================
def make_even(n):
    return n if n % 2 == 0 else n + 1

def pad_image(img, target_h, target_w):
    h, w = img.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                  mode='constant', constant_values=255)

video_count = 0
for dataset in ['train', 'test']:
    for sample in samples:
        png_files = [
            f'3D_invasion_plots_corrected/{dataset}_sample{sample}_t{tp:02d}.png'
            for tp in timepoints
            if os.path.exists(f'3D_invasion_plots_corrected/{dataset}_sample{sample}_t{tp:02d}.png')
        ]
        if not png_files:
            continue

        png_files.sort(key=lambda x: int(x.split('_t')[-1].split('.png')[0]))
        frames = [imageio.imread(p) for p in png_files]

        max_h = max(frame.shape[0] for frame in frames)
        max_w = max(frame.shape[1] for frame in frames)
        max_h_even = make_even(max_h)
        max_w_even = make_even(max_w)

        frames = [pad_image(frame, max_h_even, max_w_even) for frame in frames]

        video_path = f'3D_invasion_videos_corrected/{dataset}_sample{sample}_invasion.mp4'
        imageio.mimsave(video_path, frames, fps=2, macro_block_size=None)

        video_count += 1
        print(f"Video {video_count} created: {video_path}")

print(f"\nTotal videos created: {video_count}")
print("Time-lapse videos saved to '3D_invasion_videos_corrected/' folder.")

# =============================================================================
# 8️⃣ Summary statistics
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

for dataset in ['train', 'test']:
    for sample in samples:
        subset = combined_df[(combined_df['dataset'] == dataset) & (combined_df['sample'] == sample)]
        if subset.empty:
            continue
        print(f"\n{dataset.upper()} - Sample {sample}:")
        print(f"Total cells: {len(subset)}")
        print(f"Mean invasion distance: {subset['invasion_distance'].mean():.2f} µm")
        print(f"Max invasion distance: {subset['invasion_distance'].max():.2f} µm")
        print(f"Leader cell fraction: {subset['is_leader'].mean() * 100:.2f}%")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE ✅")
print("=" * 80)
print("Output saved here:")
print(" - 3D_invasion_plots_corrected/")
print(" - 3D_invasion_videos_corrected/")
