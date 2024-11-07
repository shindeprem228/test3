
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('sales_data_sample.csv', encoding='ISO-8859-1')

print(data.head())

 
print("Missing values in the dataset:") 
print(data.isnull().sum())

features = data.select_dtypes(include=[np.number])

features.fillna(features.mean(), inplace=True)


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

 
inertia = []
K = range(1, 11) # Test 
for k in K:
    kmeans = KMeans( n_clusters=k,random_state=42) 
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

 
plt.figure(figsize=(10, 6)) 
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method for Optimal k') 
plt.xlabel('Number of Clusters (k)') 
plt.ylabel('Inertia')
plt.xticks(K) 
plt.grid() 
plt.show()
 
optimal_k = 3 

kmeans = KMeans(n_clusters=optimal_k, random_state=42) 
data['Cluster'] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(10, 6))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=data['Cluster'], cmap='viridis') 
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2') 
plt.colorbar(label='Cluster') 
plt.grid()
plt.show()
