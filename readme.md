# Lloyd's algorithm

This repository contains a C++ implementation of Lloyd's algorithm. The core algorithm is written in C++, while Python is used to call the C++ function and visualize results using `pybind11`. 

Lloyd's algorithm is a iterative method used to perform K-means clustering. Given sets $C_1, \dots, C_K$ containing the indices of each cluster, the K-means algorithm tries to minimize the following objective:

$$
\min_{C_1, \dots, C_K} {
\sum_{k=1}^{K} \frac{1}{|C_k|} \sum_{i, i' \in C_k} \sum_{j=1}^{p} (x_{ij} - x_{i'j})^2
}.
$$

This optimization problem we would like to solve is very hard. Therefore, we can use **Lloyd's algorithm**, which iteratively updates cluster assignments and centroids and **guarantees convergence to a local optimum** of the objective function. We can formulate the objective in terms of cluster centroids as

$$
\min_{C_1, \dots, C_K} \sum_{k=1}^{K} \sum_{i \in C_k} \| x_i - \mu_k \|^2,
$$

where $\mu_k$ is the centroid (mean) of points in cluster $C_k$.

## Installation instructions

1. Install the package:

```bash
python3 -m install .
```

2. Run the main script:

```bash
python3 main.py
```  


