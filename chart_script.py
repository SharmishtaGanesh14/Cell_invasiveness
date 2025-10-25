import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from plotly.subplots import make_subplots

# Load the data
df = pd.read_csv('segmentation_results/full_segmentation_features.csv')

# Create derived features
df['intensity_range'] = df['max_intensity'] - df['min_intensity']
df['shape_irregularity'] = df['compactness']  # Using compactness as proxy
df['elongation_proxy'] = df['equivalent_diameter_area'] / np.sqrt(df['area'])  # Aspect ratio proxy

# Use compactness as invasiveness proxy (higher compactness = more invasive)
df['invasiveness_index'] = df['compactness']

# Select the 8 morphological features mentioned
features = ['area', 'volume', 'compactness', 'extent', 'shape_irregularity', 
           'elongation_proxy', 'intensity_range', 'mean_intensity']

# Prepare data for PCA
X = df[features].copy()
X = X.dropna()  # Remove any NaN values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Get corresponding invasiveness values for valid rows
invasiveness_values = df.loc[X.index, 'invasiveness_index'].values

# Create subplot figure
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["PCA Biplot", "Feature Import"],
    column_widths=[0.65, 0.35],
    horizontal_spacing=0.1
)

# PANEL 1: PCA Biplot
# Add scatter plot points
scatter = go.Scatter(
    x=X_pca[:, 0], 
    y=X_pca[:, 1],
    mode='markers',
    marker=dict(
        color=invasiveness_values,
        colorscale='RdBu_r',
        size=4,
        showscale=True,
        colorbar=dict(
            title="Invasive Idx",
            x=0.6
        )
    ),
    name='Samples',
    showlegend=False
)
fig.add_trace(scatter, row=1, col=1)

# Add feature loading arrows
loadings = pca.components_.T
feature_labels = ['area', 'volume', 'compact', 'extent', 'shape_irreg', 
                 'elongation', 'intens_range', 'mean_intens']

# Scale arrows for visibility
arrow_scale = 3.0

for i, feature in enumerate(feature_labels):
    # Arrow line
    fig.add_trace(go.Scatter(
        x=[0, loadings[i, 0] * arrow_scale],
        y=[0, loadings[i, 1] * arrow_scale],
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(color='red', size=6, symbol='arrow-up'),
        opacity=0.7,
        showlegend=False,
        name=feature,
        text=[None, feature],
        textposition='top center'
    ), row=1, col=1)

# PANEL 2: Feature Importance Bar Chart
pc1_loadings = np.abs(pca.components_[0])
pc2_loadings = np.abs(pca.components_[1])

# Sort by PC1 magnitude
sort_idx = np.argsort(pc1_loadings)[::-1]
sorted_features = [feature_labels[i] for i in sort_idx]
sorted_pc1 = pc1_loadings[sort_idx]
sorted_pc2 = pc2_loadings[sort_idx]

# Add PC1 bars
fig.add_trace(go.Bar(
    y=sorted_features,
    x=sorted_pc1,
    orientation='h',
    name='PC1',
    marker_color='#1FB8CD',
    offsetgroup=1
), row=1, col=2)

# Add PC2 bars
fig.add_trace(go.Bar(
    y=sorted_features,
    x=sorted_pc2,
    orientation='h',
    name='PC2',
    marker_color='#D2BA4C',
    offsetgroup=2
), row=1, col=2)

# Update layout
variance_explained = pca.explained_variance_ratio_
fig.update_layout(
    title="PCA Analysis of Cell Features",
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update x-axis for biplot
fig.update_xaxes(
    title=f"PC1 ({variance_explained[0]:.1%})",
    showgrid=True,
    row=1, col=1
)

# Update y-axis for biplot  
fig.update_yaxes(
    title=f"PC2 ({variance_explained[1]:.1%})",
    showgrid=True,
    row=1, col=1
)

# Update x-axis for bar chart
fig.update_xaxes(
    title="Abs Loading",
    showgrid=True,
    row=1, col=2
)

# Update y-axis for bar chart
fig.update_yaxes(
    title="Features",
    showgrid=True,
    row=1, col=2
)

# Save the figure
fig.write_image("pca_biplot.png")
fig.write_image("pca_biplot.svg", format="svg")

print(f"PCA explained variance: PC1: {variance_explained[0]:.1%}, PC2: {variance_explained[1]:.1%}")
print("Chart saved successfully!")