import numpy as np
from sklearn.datasets import make_blobs
from MyKMeans import  MyKMeans
# Generate synthetic data using sklearn.datasers.make_blobs 
X,_=make_blobs(n_samples=300, centers=3, random_state=42)
# Create a K-Means instance with 3 clusters
# X = [[1,4],[1,3],[2,3],[1,2],[6,5],[5,4],[6,4],[7,4],[6,3],]
X = np.array(X)
# kmeans = MyKMeans(n_clusters=2)
kmeans = MyKMeans(n_clusters=3)
#Fit the model to data
kmeans.fit(X)
#Get clusters assignment for each adta point
labels = kmeans._assign_labels(X)
print("Cluster Assignments", labels)
print("Final Centroids", kmeans.centroids)
print()
