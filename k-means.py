import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("FoodComposition-Original.csv")

# Select relevant features
#features = data[['Food Profile ID', 'Nitrogen Factor', 'Fat Factor', 'Specific Gravity', 'Classification']]
features = data[['Nitrogen Factor', 'Fat Factor', 'Specific Gravity', 'Classification']]

# Encode categorical features
label_encoder = LabelEncoder()
features.loc[:, 'Classification'] = label_encoder.fit_transform(features['Classification'])
#features['Food Name'] = label_encoder.fit_transform(features['Food Name'])
#features['Food Description'] = label_encoder.fit_transform(features['Food Description'])
#features['Sampling Details'] = label_encoder.fit_transform(features['Sampling Details'])

# Preprocess the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the Elbow Method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

# K-Means with Random Initialization
kmeans_random = KMeans(n_clusters=3, init='random', random_state=42)
kmeans_random.fit(scaled_features)
labels_random = 3

# K-Means with Systematic Initialization
# Select the first K samples as initial centroids
initial_centroids = scaled_features[:3]

# Create a K-Means model with the initial centroids
kmeans_systematic = KMeans(n_clusters=3, init=initial_centroids, random_state=42)
kmeans_systematic.fit(scaled_features)
labels_systematic = kmeans_systematic.labels_

# K-Means with custom initialization (you can choose random or systematic)
def k_means_custom(X, K, init_method='random'):
    if init_method == 'random':
        # Random initialization
        np.random.seed(42)
        centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    elif init_method == 'systematic':
        # Systematic initialization
        centroids = X[:K]
    else:
        raise ValueError("Invalid initialization method")

    for _ in range(100):  # Adjust the number of iterations as needed
        # Assign data points to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids  

        for i in range(K):
            centroids[i] = np.mean(X[labels == i], axis=0)

    return labels, centroids  


# Choose initialization method (random or systematic)
init_method = 'random'  # Or 'systematic'

# Apply K-Means with the chosen initialization
labels, centroids = k_means_custom(scaled_features, optimal_num_clusters, init_method)


# Plot the SSE values
plt.plot(range(1, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Choose the optimal number of clusters based on the elbow plot
optimal_num_clusters = 3  # Adjust based on the elbow plot

# Create the K-Means model with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', random_state=42)

# Fit the model to the data
kmeans.fit(scaled_features)

# Predict cluster labels
labels = kmeans.labels_

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(scaled_features, labels)
print("Silhouette Score:", silhouette_avg)

# Add cluster labels to the original DataFrame
data['Cluster'] = labels

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Nitrogen Factor', y='Fat Factor', hue='Cluster', data=data, palette='viridis')
plt.title('K-Means Clustering Visualization')
plt.xlabel('Nitrogen Factor')
plt.ylabel('Fat Factor')
plt.show()

# Visualizing Centroids With Clusters
plt.figure(figsize=(10, 8))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='*', c='black', label='Centroids')
plt.title('K-Means Clustering Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Analyze the clusters
#print(data.groupby('Cluster').describe())
