import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-whitegrid')
output_dir = ""
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("../../segmentation_results/full_segmentation_features.csv")
df["timepoint"] = df["timepoint"].astype(int)

# ----- Fixed invasion center per sample (t=0), with voxel scaling -----
voxel = np.array([1.242, 1.242, 6.0])
centers = (
    df[df["timepoint"] == df["timepoint"].min()]
    .groupby("subfolder")[["centroid-0", "centroid-1", "centroid-2"]]
    .mean()
    .rename(columns=lambda c: f"center_{c.split('-')[1]}")
)
df = df.merge(centers, left_on="subfolder", right_index=True, how="left")

def distance_from_fixed_center(row):
    scaled_coords = row[["centroid-0", "centroid-1", "centroid-2"]].values * voxel
    scaled_center = row[["center_0", "center_1", "center_2"]].values * voxel
    return np.linalg.norm(scaled_coords - scaled_center)

df["distance_from_center"] = df.apply(distance_from_fixed_center, axis=1)

# ----- Leader fraction (using 90th percentile) -----
leader_results = []
for (sample, tp), subdf in df.groupby(["subfolder", "timepoint"]):
    if len(subdf) == 0:
        continue
    threshold = np.percentile(subdf["distance_from_center"], 90)
    n_leaders = np.sum(subdf["distance_from_center"] >= threshold)
    leader_fraction = n_leaders / len(subdf)
    leader_results.append([sample, tp, leader_fraction])

leader_df = pd.DataFrame(leader_results, columns=["sample", "timepoint", "leader_fraction"])

plt.figure(figsize=(8,5))
sns.lineplot(data=leader_df, x="timepoint", y="leader_fraction", hue="sample", marker="o")
plt.title("Leader Cell Fraction over Time")
plt.xlabel("Timepoint")
plt.ylabel("Fraction of Leader Cells (>90th percentile)")
plt.legend(title="Sample")
plt.tight_layout()
plt.savefig(f"{output_dir}/leader_fraction_over_time.png", dpi=300)
plt.close()

# ----- Morphology–Distance Correlation -----
corr_results = []
for (sample, tp), subdf in df.groupby(["subfolder", "timepoint"]):
    if len(subdf) > 5:
        corr = subdf["compactness"].corr(subdf["distance_from_center"])
        corr_results.append([sample, tp, corr])

corr_df = pd.DataFrame(corr_results, columns=["sample", "timepoint", "compactness_distance_corr"])

plt.figure(figsize=(8,5))
sns.lineplot(data=corr_df, x="timepoint", y="compactness_distance_corr", hue="sample", marker="o")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Correlation between Cell Compactness and Distance")
plt.xlabel("Timepoint")
plt.ylabel("Pearson r (Compactness vs. Distance)")
plt.legend(title="Sample")
plt.tight_layout()
plt.savefig(f"{output_dir}/morphology_distance_correlation.png", dpi=300)
plt.close()

# ----- Anisotropy of Invasion (Ellipticity of spatial spread) -----
anisotropy_results = []
for (sample, tp), subdf in df.groupby(["subfolder", "timepoint"]):
    coords = subdf[["centroid-0", "centroid-1", "centroid-2"]].values * voxel
    if len(coords) > 1:
        cov = np.cov(coords.T)
        eigvals = np.linalg.eigvalsh(cov)
        anisotropy = eigvals.max() / eigvals.min() if eigvals.min() > 0 else np.nan
        anisotropy_results.append([sample, tp, anisotropy])
    else:
        anisotropy_results.append([sample, tp, np.nan])

aniso_df = pd.DataFrame(anisotropy_results, columns=["sample", "timepoint", "anisotropy"])

plt.figure(figsize=(8,5))
sns.lineplot(data=aniso_df, x="timepoint", y="anisotropy", hue="sample", marker="o")
plt.title("Invasion Anisotropy over Time")
plt.xlabel("Timepoint")
plt.ylabel("Ellipticity Ratio (max/min spread)")
plt.legend(title="Sample")
plt.tight_layout()
plt.savefig(f"{output_dir}/anisotropy_over_time.png", dpi=300)
plt.close()

# ----- Invasion Volume vs Cell Density (physical units) -----
volume_density_results = []
for (sample, tp), subdf in df.groupby(["subfolder", "timepoint"]):
    if len(subdf) == 0:
        continue
    max_distance = subdf["distance_from_center"].max()
    invasion_volume = (4/3) * np.pi * (max_distance ** 3)
    density = len(subdf) / invasion_volume if invasion_volume > 0 else 0
    volume_density_results.append([sample, tp, invasion_volume, density])

vd_df = pd.DataFrame(volume_density_results, columns=["sample", "timepoint", "invasion_volume", "cell_density"])

fig, ax1 = plt.subplots(figsize=(8,5))
sns.lineplot(data=vd_df, x="timepoint", y="invasion_volume", hue="sample", marker="o", ax=ax1)
ax1.set_ylabel("Invasion Volume (µm³)", color='tab:blue')
ax1.set_xlabel("Timepoint")
ax2 = ax1.twinx()
sns.lineplot(data=vd_df, x="timepoint", y="cell_density", hue="sample", marker="x", linestyle="--", ax=ax2, legend=False)
ax2.set_ylabel("Cell Density (cells/µm³)", color='tab:red')
plt.title("Invasion Volume and Cell Density over Time")
plt.tight_layout()
plt.savefig(f"{output_dir}/invasion_volume_density.png", dpi=300)
plt.close()

# ----- Summary Output -----
leader_df.to_csv(f"{output_dir}/leader_fraction.csv", index=False)
corr_df.to_csv(f"{output_dir}/compactness_distance_corr.csv", index=False)
aniso_df.to_csv(f"{output_dir}/anisotropy.csv", index=False)
vd_df.to_csv(f"{output_dir}/volume_density.csv", index=False)

print("✅ All invasion analysis plots and CSVs saved in:", output_dir)
