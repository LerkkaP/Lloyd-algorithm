#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>

int getClosestCluster(std::array<double, 2> point, const std::vector<std::array<double, 2>> &centroids);
std::array<double, 2> getNewCentroid(const std::vector<std::array<double,2>> &cluster);
std::vector<std::vector<std::array<double, 2>>> kMeansClustering(const int k, const std::vector<std::array<double, 2>> &data);
std::vector<std::array<double, 2>> initializeCentroids(const int k, const std::vector<std::array<double, 2>> &data);

std::vector<std::vector<std::array<double, 2>>> kMeansClustering(const int k, const std::vector<std::array<double, 2>> &data)
{   
    std::vector<std::array<double, 2>> centroids{initializeCentroids(k, data)};
    std::vector<std::vector<std::array<double,2>>> clusters(k);
    
    int iterations { };
    constexpr int max_iters { 2 };

    while (iterations <= max_iters) {
        // Start with empty clusters at each iteration
        clusters = std::vector<std::vector<std::array<double,2>>>(k);

        // Assign point to nearest cluster
        for (std::array<double, 2> point : data) {
            int closestCluster = getClosestCluster(point, centroids);
            clusters[closestCluster].push_back(point);
        }

        // Update centroids
        for (int i = 0; i < k; ++i) {
            centroids[i] = getNewCentroid(clusters[i]);
        }
        // Compute objective here
        double loss = objective(clusters);
        ++iterations;
    }
    return clusters;
}

int getClosestCluster(std::array<double, 2> point, const std::vector<std::array<double, 2>> &centroids) 
{
    double minDistance = std::numeric_limits<double>::max();
    int closestCluster { -1 };

    for (int i = 0; i < centroids.size(); ++i) {
        double distance = sqrt(pow(point[0] - centroids[i][0], 2) + pow(point[1] - centroids[i][1], 2));
        if (distance < minDistance) {
            minDistance = distance;
            closestCluster = i;
        }
    }
    return closestCluster;
}

std::array<double, 2> getNewCentroid(const std::vector<std::array<double,2>> &cluster) {
    std::array<double, 2> newMean{0.0, 0.0};      
    if (cluster.empty()) return newMean;         

    for (auto &point : cluster) {                 
        newMean[0] += point[0];                  
        newMean[1] += point[1];                  
    }

    newMean[0] /= cluster.size();                
    newMean[1] /= cluster.size();                
    return newMean;                               
}

std::vector<std::array<double, 2>> initializeCentroids(const int k, const std::vector<std::array<double,2>> &data)
{
    std::vector<std::array<double, 2>> centroids{};
    std::unordered_set<int> used_indices{};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    while (centroids.size() < k) {
        int index = dis(gen);
        if (used_indices.insert(index).second) {
            centroids.push_back(data[index]);
        }
    }
    return centroids;
}

double objective(const std::vector<std::vector<std::array<double,2>>> &clusters) {
    double loss = 0;

    for (const auto &cluster: clusters) {
        auto centroid = getNewCentroid(cluster);

        for (const auto &point : cluster) {
            loss += pow(point[0] - centroid[0], 2)
                 +  pow(point[1] - centroid[1], 2);
        }
    }
    return loss;
}

PYBIND11_MODULE(kmeans, m) {
    m.def("kMeansClustering", &kMeansClustering, "Run k-means clustering",
          pybind11::arg("k"), pybind11::arg("data"));
}