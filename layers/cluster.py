# coding : utf-8
# Author : Yuxiang Zeng

import random
import math


def euclidean_distance(p1, p2):
    """计算两点间欧氏距离"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))


def initialize_centroids(points, k):
    """随机初始化k个质心"""
    return random.sample(points, k)


def assign_points_to_clusters(points, centroids):
    """根据当前质心分配每个点"""
    clusters = [[] for _ in centroids]
    for point in points:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        min_index = distances.index(min(distances))
        clusters[min_index].append(point)
    return clusters


def calculate_new_centroids(clusters):
    """根据每个簇的点，重新计算质心"""
    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:  # 避免空簇
            new_centroids.append([0 for _ in range(len(clusters[0][0]))])
        else:
            centroid = []
            for dim in range(len(cluster[0])):
                coord_sum = sum(point[dim] for point in cluster)
                centroid.append(coord_sum / len(cluster))
            new_centroids.append(centroid)
    return new_centroids


def kmeans(points, k, max_iterations=100, tolerance=1e-4):
    """K-Means主程序"""
    centroids = initialize_centroids(points, k)

    for _ in range(max_iterations):
        clusters = assign_points_to_clusters(points, centroids)
        new_centroids = calculate_new_centroids(clusters)

        shift = sum(euclidean_distance(c, nc) for c, nc in zip(centroids, new_centroids))
        if shift < tolerance:
            break
        centroids = new_centroids

    return centroids, clusters


# 示例
if __name__ == "__main__":
    data = [
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0]
    ]
    k = 2
    centroids, clusters = kmeans(data, k)
    print("Centroids:", centroids)
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx + 1}: {cluster}")