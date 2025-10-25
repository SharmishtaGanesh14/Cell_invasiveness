"""
PCA and Clustering Analysis for Invasion Metrics
This script performs dimensionality reduction and clustering to identify
natural groupings in invasion behavior.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway

print("="*80)
print("PCA AND CLUSTERING ANALYSIS")
print("="*80)

# Read data
metrics = pd.read_csv('../enhanced_invasion_metrics.csv')

# Select features
feature_cols = [
    'mean_radius', 'max_radius', 'sd_radius',
    'p90_radius', 'n_leader_cells', 'mean_leader_radius',
    'radius_skewness', 'radius_kurtosis', 'dispersion_index',
    'mean_nn_dist', 'median_nn_dist',
    'mean_volume', 'mean_compactness', 'mean_extent',
    'cell_count', 'cell_density'
]

print(f"\n[1] Selected {len(feature_cols)} features")

# Standardize and perform PCA
X = metrics[feature_cols].values
X_scaled = StandardScaler().fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\n[2] PCA Results:")
print(f"    PC1: {explained_var[0]*100:.2f}% variance")
print(f"    PC2: {explained_var[1]*100:.2f}% variance")
print(f"    PC1+PC2: {cumulative_var[1]*100:.2f}% total variance")

# FIX: Assign PCs before statistical tests
metrics['PC1'] = X_pca[:, 0]
metrics['PC2'] = X_pca[:, 1]
if X_pca.shape[1] > 2:
    metrics['PC3'] = X_pca[:, 2]

# Add grouping variables
metrics['condition'] = metrics['dataset'] + '_S' + metrics['sample'].astype(str)
metrics['time_period'] = metrics['timepoint'].apply(
    lambda x: 'Early (t0-3)' if x <= 3 else 'Mid (t4-7)' if x <= 7 else 'Late (t8-11)'
)

# K-means clustering
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
metrics['cluster'] = cluster_labels

print(f"\n[3] K-Means Clustering (k={optimal_k}):")
for cluster_id in range(optimal_k):
    cluster_data = metrics[metrics['cluster'] == cluster_id]
    print(f"\n    Cluster {cluster_id}: {len(cluster_data)} observations")
    cond_dist = cluster_data['condition'].value_counts()
    print(f"      Conditions: {dict(cond_dist)}")
    print(f"      Mean invasion: {cluster_data['mean_radius'].mean():.1f} μm")

# Hierarchical clustering
linkage_matrix = linkage(X_scaled, method='ward')

# Statistical tests
conditions = metrics['condition'].unique()
pc1_by_condition = [metrics[metrics['condition'] == c]['PC1'].values for c in conditions]
pc2_by_condition = [metrics[metrics['condition'] == c]['PC2'].values for c in conditions]

f_stat_pc1, p_val_pc1 = f_oneway(*pc1_by_condition)
f_stat_pc2, p_val_pc2 = f_oneway(*pc2_by_condition)

print(f"\n[4] Statistical Separation (ANOVA):")
print(f"    PC1 by Condition: p={p_val_pc1:.2e} ({'SIGNIFICANT' if p_val_pc1 < 0.05 else 'NOT SIG'})")
print(f"    PC2 by Condition: p={p_val_pc2:.2e} ({'SIGNIFICANT' if p_val_pc2 < 0.05 else 'NOT SIG'})")

# Save results
pca_results = metrics[['dataset', 'sample', 'timepoint', 'condition', 
                       'PC1', 'PC2', 'PC3', 'cluster']]
pca_results.to_csv('pca_clustering_results.csv', index=False)

# Create visualizations
fig = plt.figure(figsize=(20, 12))

# Colors
condition_colors = {'test_S1': '#2ca02c', 'test_S2': '#d62728', 
                   'train_S1': '#1f77b4', 'train_S2': '#ff7f0e'}

# Plot 1: PCA by Condition
ax1 = plt.subplot(2, 3, 1)
for cond in metrics['condition'].unique():
    mask = metrics['condition'] == cond
    ax1.scatter(metrics.loc[mask, 'PC1'], metrics.loc[mask, 'PC2'], 
               c=condition_colors[cond], label=cond, s=100, alpha=0.7, edgecolors='black')
ax1.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontweight='bold')
ax1.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontweight='bold')
ax1.set_title('PCA: By Condition', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: PCA by Cluster
ax2 = plt.subplot(2, 3, 2)
scatter = ax2.scatter(metrics['PC1'], metrics['PC2'], 
                     c=metrics['cluster'], cmap='viridis', 
                     s=100, alpha=0.7, edgecolors='black')
ax2.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontweight='bold')
ax2.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontweight='bold')
ax2.set_title('PCA: K-Means Clusters', fontweight='bold')
plt.colorbar(scatter, ax=ax2)
ax2.grid(True, alpha=0.3)

# Plot 3: Scree Plot
ax3 = plt.subplot(2, 3, 3)
ax3.plot(range(1, 11), explained_var[:10]*100, 'bo-', linewidth=2)
ax3.plot(range(1, 11), cumulative_var[:10]*100, 'ro--', linewidth=2)
ax3.set_xlabel('Principal Component', fontweight='bold')
ax3.set_ylabel('Variance (%)', fontweight='bold')
ax3.set_title('Scree Plot', fontweight='bold')
ax3.legend(['Individual', 'Cumulative'])
ax3.grid(True, alpha=0.3)

# Plot 4: Loadings Heatmap
ax4 = plt.subplot(2, 3, 4)
loadings = pca.components_[:3, :].T
im = ax4.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.6)
ax4.set_yticks(range(len(feature_cols)))
ax4.set_yticklabels(feature_cols, fontsize=8)
ax4.set_xticks([0, 1, 2])
ax4.set_xticklabels(['PC1', 'PC2', 'PC3'])
ax4.set_title('Feature Loadings', fontweight='bold')
plt.colorbar(im, ax=ax4)

# Plot 5: Dendrogram
ax5 = plt.subplot(2, 3, 5)
labels = [f"{row['condition']}_t{row['timepoint']:02d}" 
          for _, row in metrics.iterrows()]
dendrogram(linkage_matrix, ax=ax5, labels=labels, leaf_font_size=5, 
          leaf_rotation=90)
ax5.set_title('Hierarchical Clustering', fontweight='bold')
ax5.tick_params(axis='x', labelsize=5)

# Plot 6: Cluster Statistics
ax6 = plt.subplot(2, 3, 6)
cluster_stats = []
for c in range(optimal_k):
    data = metrics[metrics['cluster'] == c]
    cluster_stats.append([
        data['mean_radius'].mean(),
        data['dispersion_index'].mean(),
        data['n_leader_cells'].mean()
    ])
cluster_stats = np.array(cluster_stats)
x = np.arange(optimal_k)
width = 0.25
ax6.bar(x - width, cluster_stats[:, 0]/100, width, label='Mean Radius (/100)')
ax6.bar(x, cluster_stats[:, 1], width, label='Dispersion')
ax6.bar(x + width, cluster_stats[:, 2], width, label='Leader Cells')
ax6.set_xlabel('Cluster', fontweight='bold')
ax6.set_ylabel('Value', fontweight='bold')
ax6.set_title('Cluster Characteristics', fontweight='bold')
ax6.set_xticks(x)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('PCA and Clustering Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_clustering_analysis.png', dpi=300, bbox_inches='tight')
print("\n[5] ✓ Saved: pca_clustering_analysis.png")
plt.close()

print("\n[6] ✓ Saved: pca_clustering_results.csv")
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
