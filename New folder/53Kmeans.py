import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

# Display the first few rows of the dataframe
print(df.head())

# Display data types of the columns
print(df.dtypes)

# Select the features for clustering (replace with actual column indices or names)
X = df.iloc[:, [3, 4]].values  # Adjust indices based on your dataset

# Calculate WCSS (within-cluster sum of squares) for different values of k
wcss = []  # within cluster sum of square
for i in range(1, 11):
    # Initialize the KMeans algorithm
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    # Append the WCSS value for each k
    wcss.append(kmeans.inertia_)

# Create a plot for the Elbow method
ks = list(range(1, 11))
plt.figure(figsize=(10, 5))
plt.plot(ks, wcss, 'bx-')
plt.title("Elbow Method")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.grid()
plt.show()

# Scaling the data for clustering
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Fit KMeans with the optimal number of clusters (change k based on elbow plot)
optimal_k = 3  # Example: Set this to the k value you determine from the elbow plot
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42)
kmeans.fit(scaled_X)

# Add cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Plotting the clusters
plt.figure(figsize=(10, 5))
plt.scatter(scaled_X[:, 0], scaled_X[:, 1], c=df['Cluster'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title(f"K-Means Clustering (k={optimal_k})")
plt.xlabel("Feature 1")  # Change to the name of your first feature
plt.ylabel("Feature 2")  # Change to the name of your second feature
plt.grid()
plt.colorbar(label='Cluster Label')
plt.show()
