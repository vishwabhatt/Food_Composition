import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv('/Food Composition.csv')
numeric_data = data[['Nitrogen Factor', 'Fat Factor', 'Specific Gravity']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
gmm = GaussianMixture(n_components=3, random_state=0)
data['Cluster'] = gmm.fit_predict(scaled_data)
sil_score = silhouette_score(scaled_data, data['Cluster'])
print(f"Silhouette Score: {sil_score:.3f}")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=data['Cluster'], palette='viridis')
plt.xlabel('Nitrogen Factor')
plt.ylabel('Fat Factor')
plt.title('Scatter Plot of Nitrogen Factor vs. Fat Factor')
plt.legend(title='Cluster')
plt.show()
pairplot_data = pd.DataFrame(scaled_data, columns=['Nitrogen Factor', 'Fat Factor', 'Specific Gravity'])
pairplot_data['Cluster'] = data['Cluster']
sns.pairplot(pairplot_data, hue='Cluster', palette='viridis', diag_kind='kde')
plt.suptitle('Pair Plot of Food Composition Features', y=1.02)
plt.show()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2], c=data['Cluster'], cmap='viridis')
ax.set_xlabel('Nitrogen Factor')
ax.set_ylabel('Fat Factor')
ax.set_zlabel('Specific Gravity')
plt.title('3D Scatter Plot of Food Composition Data')
plt.colorbar(scatter, label='Cluster')
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
features = ['Nitrogen Factor', 'Fat Factor', 'Specific Gravity']
for i, feature in enumerate(features):
    for cluster in np.unique(data['Cluster']):
        sns.kdeplot(scaled_data[data['Cluster'] == cluster][:, i], ax=axes[i], label=f'Cluster {cluster}', fill=True)
    axes[i].set_title(f'Distribution of {feature} by Cluster')
    axes[i].set_xlabel(feature)
    axes[i].legend()
plt.suptitle('KDE Plots for Each Feature by Cluster', y=1.02)
plt.tight_layout()
plt.show()
