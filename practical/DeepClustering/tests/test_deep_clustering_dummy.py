
from practical.DeepClustering.deep_clustering_dummy import hello_world, minibatch_k_means
from sklearn.cluster import KMeans
import torch


def test_hello_world_clustering():
    assert hello_world("Deep Clustering") == "Hello Deep Clustering"


def test_minibatch_k_means():
    data = torch.rand(20, 2)
    cluster = 3

    minibatch_k_means_centers = minibatch_k_means(cluster, 10, 100, data)

    sklearn_k_means = KMeans(cluster)
    sklearn_k_means.fit(data)
    sklearn_k_means_centers = torch.tensor(sklearn_k_means.cluster_centers_, dtype=torch.float)

    assert torch.allclose(minibatch_k_means_centers, sklearn_k_means_centers, atol=1)


test_minibatch_k_means()