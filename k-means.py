import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("FoodComposition-Original.csv")

# Select relevant features
features = data[['Nitrogen Factor', 'Fat Factor', 'Specific Gravity', 'Classification']]

# Preprocess the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,
 init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')

plt.ylabel('WCSS')
plt.show()

# Based on the elbow method...
n_clusters = 3

# Create a K-Means clustering model
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
kmeans.fit(scaled_features)

# Predict cluster labels
labels = kmeans.labels_

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(scaled_features, labels)
print("Silhouette Score:", silhouette_avg)

# Add cluster labels to the original DataFrame
data['Cluster'] = labels

# Analyze the clusters
print(data.groupby('Cluster').describe())
