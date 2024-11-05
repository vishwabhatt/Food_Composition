# Step 1: Import necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 2: Load the dataset
data = pd.read_csv('/content/Clustered_Food_Data.csv') 
print("Dataset loaded successfully. Here are the first few rows:\n", data.head())

# Step 3: Define the DBSCAN model pipeline and fit to data
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('dbscan', DBSCAN(eps=0.5, min_samples=5))  # Adjust `eps` and `min_samples` as needed
])

# Transform 'Food Name' column using TF-IDF and apply DBSCAN
tfidf_matrix = pipeline.named_steps['tfidf'].fit_transform(data['Food Name'])
labels = pipeline.named_steps['dbscan'].fit_predict(tfidf_matrix)
data['Cluster'] = labels
print("\nClusters assigned to each food item.")

# Step 4: Calculate Silhouette Score and Davies-Bouldin Index 
if -1 not in labels:
    silhouette_avg = silhouette_score(tfidf_matrix, labels)
    print(f"\nSilhouette Score: {silhouette_avg:.3f}")
else:
    print("\nSilhouette Score is not calculated due to the presence of noise points.")

# Calculate Davies-Bouldin Index
davies_bouldin_avg = davies_bouldin_score(tfidf_matrix.toarray(), labels)
print(f"Davies-Bouldin Index: {davies_bouldin_avg:.3f}")

# Step 5: Cluster Size Distribution Bar Chart 
cluster_counts = data[data['Cluster'] != -1]['Cluster'].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
plt.title("Cluster Size Distribution (Excluding Noise)")
plt.xlabel("Cluster Label")
plt.ylabel("Number of Items")
plt.show()

# Step 6: Cluster Size Distribution Pie Chart (excluding noise points)
plt.figure(figsize=(10, 8))
plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(cluster_counts)))
plt.title("Cluster Size Distribution (Excluding Noise)")
plt.show()

# Step 7: t-SNE Visualization of Clusters
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
