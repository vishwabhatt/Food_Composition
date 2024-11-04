import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv('/Food Composition.csv')

# Select four numeric columns for clustering
numeric_data = data[['Nitrogen Factor', 'Fat Factor', 'Specific Gravity', 'Classification']]

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Define and fit the GMM model with 4 clusters
gmm = GaussianMixture(n_components=4, random_state=0)
data['Cluster'] = gmm.fit_predict(scaled_data)

# Plot 1: Scatter plot of Nitrogen Factor vs. Fat Factor, colored by Cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=data['Cluster'], palette='viridis')
plt.xlabel('Nitrogen Factor')
plt.ylabel('Fat Factor')
plt.title('Scatter Plot of Nitrogen Factor vs. Fat Factor')
plt.legend(title='Cluster')
plt.show()

# Plot 2: Pair plot of all numeric features, with clusters highlighted
pairplot_data = pd.DataFrame(scaled_data, columns=['Nitrogen Factor', 'Fat Factor', 'Specific Gravity', 'Classification'])
pairplot_data['Cluster'] = data['Cluster']
sns.pairplot(pairplot_data, hue='Cluster', palette='viridis', diag_kind='kde')
plt.suptitle('Pair Plot of Food Composition Features', y=1.02)
plt.show()

# Plot 3: 3D Scatter plot of three features with clusters in 3D space
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2], c=data['Cluster'], cmap='viridis')
ax.set_xlabel('Nitrogen Factor')
ax.set_ylabel('Fat Factor')
ax.set_zlabel('Specific Gravity')
plt.title('3D Scatter Plot of Food Composition Data')
plt.colorbar(scatter, label='Cluster')
plt.show()


# Plot 5: KDE plot for each feature by cluster
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
features = ['Nitrogen Factor', 'Fat Factor', 'Specific Gravity', 'Classification']
for i, feature in enumerate(features):
    for cluster in np.unique(data['Cluster']):
        sns.kdeplot(scaled_data[data['Cluster'] == cluster][:, i], ax=axes[i], label=f'Cluster {cluster}', fill=True)
    axes[i].set_title(f'Distribution of {feature} by Cluster')
    axes[i].set_xlabel(feature)
    axes[i].legend()

plt.suptitle('KDE Plots for Each Feature by Cluster', y=1.02)
plt.tight_layout()
plt.show()
