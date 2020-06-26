import matplotlib.pyplot as plt
import matplotlib.marker as marker

# Create some data
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2,
    centers=4, cluster_std=1.8, random_state=101)

plt.scatter(data[0][:,0], data[0][:,1], c=data[1],
    cmap='rainbow', edgecolor='black')


# Creating Clusters
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])

# Plot original and predict clusters
def kmeans_vs_original():
    X1 = data[0][:,0]
    X2 = data[0][:,1]
    original_labels = data[1]

    plot_styles = dict(cmap='rainbow', edgecolor='black')

    _f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
    ax1.set_title('K Means')
    ax1.scatter(X1, X2, c=kmeans.labels_, **plot_styles)
    ax2.set_title('Original')
    ax2.scatter(X1, X2, c=original_labels, **plot_styles)
