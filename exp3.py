# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Assuming the cleaned dataset is saved as 'cleaned_data.csv'
data = pd.read_csv('cleaned_data.csv')
print(data.head())  # Display the first few rows

# Normalize the data using StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Split the data into training and testing subsets (if applicable)
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(scaled_data, test_size=0.3, random_state=42)


# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters as needed
kmeans.fit(X_train)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust hyperparameters as needed
dbscan.fit(X_train)

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)  # Adjust n_clusters as needed
agglo.fit(X_train)


# Plot K-Means clustering
plt.figure(figsize=(10, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()

# Similar plots can be generated for DBSCAN and Agglomerative clustering
# Evaluate K-Means
kmeans_silhouette = silhouette_score(X_train, kmeans.labels_)
kmeans_db_index = davies_bouldin_score(X_train, kmeans.labels_)
print(f'K-Means Silhouette Score: {kmeans_silhouette}')
print(f'K-Means Davies-Bouldin Index: {kmeans_db_index}')

# Evaluate DBSCAN
dbscan_silhouette = silhouette_score(X_train, dbscan.labels_)
dbscan_db_index = davies_bouldin_score(X_train, dbscan.labels_)
print(f'DBSCAN Silhouette Score: {dbscan_silhouette}')
print(f'DBSCAN Davies-Bouldin Index: {dbscan_db_index}')

# Evaluate Agglomerative Clustering
agglo_silhouette = silhouette_score(X_train, agglo.labels_)
agglo_db_index = davies_bouldin_score(X_train, agglo.labels_)
print(f'Agglomerative Clustering Silhouette Score: {agglo_silhouette}')
print(f'Agglomerative Clustering Davies-Bouldin Index: {agglo_db_index}')


# Summarize and compare metrics
results = {
    "Algorithm": ["K-Means", "DBSCAN", "Agglomerative"],
    "Silhouette Score": [kmeans_silhouette, dbscan_silhouette, agglo_silhouette],
    "Davies-Bouldin Index": [kmeans_db_index, dbscan_db_index, agglo_db_index]
}

results_df = pd.DataFrame(results)
print(results_df)
