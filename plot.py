import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_clusters_2d(k, clusters):
    sns.set_theme()
    plt.figure(figsize=(6,6))
    
    colors = sns.color_palette("viridis", k)
    for i in range(k):
        cluster_points = np.array(clusters[i])
        plt.scatter(cluster_points[:,0], cluster_points[:,1], color=colors[i], label=f"Cluster {i}", alpha=0.7)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("2D K-Means Clusters")
    plt.legend()
    plt.grid(True)
    plt.savefig("clusters.png")

def plot_loss(losses):
    sns.set_theme()
    plt.figure(figsize=(6,6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss function")
    plt.grid(True)
    plt.savefig("loss.png")
