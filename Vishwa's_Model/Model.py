
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
data = pd.read_csv('/content/Food Composition.csv')  # Adjust this path to your file
print("Dataset loaded successfully. Here are the first few rows:\n", data.head())

# Load the pre-trained clustering pipeline model
pipeline = joblib.load('/content/food_name_clustering_pipeline.joblib')
print("\nModel pipeline loaded. Predicting clusters...")

# Predict clusters for the dataset
labels = pipeline.predict(data['Food Name'])
data['Cluster'] = labels
print("\nClusters assigned to each food item.")

# Calculate and display Silhouette Score and Davies-Bouldin Index
silhouette_avg = silhouette_score(pipeline.named_steps['tfidf'].transform(data['Food Name']), labels)
davies_bouldin_avg = davies_bouldin_score(pipeline.named_steps['tfidf'].transform(data['Food Name']).toarray(), labels)
print(f"\nSilhouette Score: {silhouette_avg:.3f}")
print("Silhouette Score measures cluster cohesion and separation. Higher scores indicate better-defined clusters.\n")
print(f"Davies-Bouldin Index: {davies_bouldin_avg:.3f}")
print("Davies-Bouldin Index reflects cluster separation; lower values indicate better separation between clusters.\n")

# Cluster Size Distribution Bar Chart
cluster_counts = data['Cluster'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title("Cluster Size Distribution")
plt.xlabel("Cluster Label")
plt.ylabel("Number of Items")
plt.show()

# Cluster Size Distribution Pie Chart
plt.figure(figsize=(10, 8))
plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(cluster_counts)))
plt.title("Cluster Size Distribution")
plt.show()

# PCA-based Scatter Plot for Cluster Visualization
# Reduce dimensions to 2D for visualization using PCA
tfidf_matrix = pipeline.named_steps['tfidf'].transform(data['Food Name'])
pca = PCA(n_components=2, random_state=42)
pca_results = pca.fit_transform(tfidf_matrix.toarray())

# Plot PCA-based cluster visualization
plt.figure(figsize=(10, 8))
sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=labels, palette='viridis', s=50, alpha=0.7)
plt.title("Cluster Distributions of Food Names (PCA-reduced TF-IDF)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster", loc="upper right")
plt.show()

# t-SNE Visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
tsne_data = pd.DataFrame(tsne_results, columns=['t-SNE1', 't-SNE2'])
tsne_data['Cluster'] = labels

plt.figure(figsize=(12, 10))
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Cluster', palette='viridis', data=tsne_data, s=50, alpha=0.7)
plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Cluster', loc='best')
plt.show()
