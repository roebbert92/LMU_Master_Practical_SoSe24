from practical.DeepClustering.robin_loebbert import MiniBatchKMeans
from sklearn.datasets import load_wine
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy
from sklearn.cluster import MiniBatchKMeans as skminibatch
import time


def test_minibatch_kmeans():
    data, labels = load_wine(return_X_y=True)
    start_time = time.perf_counter()
    kmeans = MiniBatchKMeans(10, 1024, 20, device="cuda")
    pred_labels = kmeans.fit(data).predict(data)
    print("my train acc: ", unsupervised_clustering_accuracy(labels, pred_labels))
    print("duration: ", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    sk_kmeans = skminibatch(
        10, init="random", batch_size=1024, max_iter=20, random_state=42
    )
    pred_labels = sk_kmeans.fit_predict(data)
    print("sklearn train_acc: ", unsupervised_clustering_accuracy(labels, pred_labels))
    print("duration: ", time.perf_counter() - start_time)
    test, test_labels = load_fmnist("test")
    print(
        "my test acc: ",
        unsupervised_clustering_accuracy(test_labels, kmeans.predict(test)),
    )
    print(
        "sklearn test acc: ",
        unsupervised_clustering_accuracy(test_labels, sk_kmeans.predict(test)),
    )
