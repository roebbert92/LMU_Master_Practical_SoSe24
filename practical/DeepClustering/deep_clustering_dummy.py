import torch


def hello_world(string: str) -> str:
    return "Hello " + string


def minibatch_k_means(k: int, size: int, iterations: int, X):
    v = torch.zeros(k)
    centroids = X[torch.randint(0, len(X), (k,))]

    for iteration in range(iterations):
        batch_index = torch.randint(0, len(X), (size,))
        batch = X[batch_index]
        distance = torch.argmin(torch.cdist(batch, centroids), dim=1)

        for index, point in enumerate(batch):
            center = distance[index]
            v[center] += 1
            learning_rate = 1 / v[center]
            centroids[center] = (1 - learning_rate) * centroids[center] + learning_rate * point

    return centroids
