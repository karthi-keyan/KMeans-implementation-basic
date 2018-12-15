# KMeans-implementation-basic
Basic KMeans implementation with visualization
Kmeans clustering is implemented using python and can be used for understanding how the K-Means clustering works amd math behind it.
It uses only following steps:
     1.Pick random points from data to cluster based on the number of cluster to be formed which are called as centiroids.
     2.for each points find the closest centiroid and assign the corresponding class to that point.
     3.find the centiroid between that point and prev centroid and update it.
     4.repeat the step 2 until all centroids converges and won't change at all. Generally sklearn lib uses 300 iterations.
Finding the number of clustering:
     Based on variance analysis.For particular cluster size variance won't change that much it is called as elbow point that cluster size is the optimal cluster size.
