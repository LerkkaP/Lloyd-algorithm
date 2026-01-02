from sklearn.datasets import make_blobs
import kmeans
from plot import plot_clusters_2d, plot_loss

k = 3
X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=1.0, random_state=42)

clusters, losses = kmeans.kMeansClustering(k, X.tolist())
plot_clusters_2d(k, clusters)
plot_loss(losses)
