import torch
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler, TensorDataset
import numpy as np
from clustpy.data import load_fmnist
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy
from sklearn.cluster import MiniBatchKMeans as skminibatch
import time


# using clustpy (evaluation) + pytorch
class MiniBatchKMeans:
    k: int
    b: int
    t: int
    c: torch.Tensor
    rng: torch.Generator
    distance_p_norm: int

    def __init__(
        self,
        k,
        mini_batch_size,
        iterations,
        distance_p_norm=2,
        seed=42,
        device="cpu",
    ) -> None:
        self.k = k
        self.b = mini_batch_size
        self.t = iterations
        self.rng = torch.random.manual_seed(seed)
        self.device = device
        self.distance_p_norm = distance_p_norm

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        if y is None:
            dataset = TensorDataset(torch.tensor(X))
        else:
            dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        dataloader = DataLoader(dataset, self.b, shuffle=True, generator=self.rng)
        # init C with k random entries from X
        random_idx = (
            torch.randint(
                high=len(X),
                size=(self.k,),
                device="cpu",
                generator=self.rng,
                requires_grad=False,
            )
            .numpy()
            .tolist()
        )
        self.c = torch.tensor(X[random_idx], device=self.device)
        # zero init per center counts
        v = torch.zeros(self.c.size(0), dtype=torch.long, device=self.device)
        # epochs / iterations
        for i in range(self.t):
            # get random minibatch
            loss_per_epoch = torch.tensor(0.0).cpu()
            for batch in dataloader:
                # foreach x in minibatch
                if y is None:
                    (x,) = batch
                else:
                    (x, _) = batch
                x = x.to(self.device)
                # get nearest center index: dist = (b, k)
                dist = torch.cdist(x, self.c, p=self.distance_p_norm)
                d = dist.argmin(1)
                # summing up the items per nearest cluster: x = (k, d)
                x_sum_per_c = torch.zeros_like(self.c, device=self.device).index_add_(
                    0, d, x
                )
                # update the center counts for this center
                v_indices, counts = d.unique(sorted=True, return_counts=True, dim=0)
                v = v.index_add_(0, v_indices, counts)
                # get per center learning rate
                eta = (1 / v).unsqueeze(1)
                eta[eta == float("inf")] = 0
                # take gradient step: self.c = (k,d), eta = (k,), batch = (b,d), d = (b,)
                loss = eta * x_sum_per_c
                loss_per_epoch += torch.sum(loss).cpu()
                self.c = (1 - eta) * self.c + loss
            print(f"loss (epoch {i}):", float(loss_per_epoch))
            if y is not None:
                y_hat = self.predict(X)
                print("train_acc: ", unsupervised_clustering_accuracy(y, y_hat))
        return self

    def predict(self, X: np.ndarray):
        # dataloader for one iteration of minibatches
        dataset = TensorDataset(torch.tensor(X))
        dataloader = DataLoader(dataset, self.b, shuffle=False)
        results = []
        for batch in dataloader:
            (x,) = batch
            x = x.to(self.device)
            results.append(
                torch.cdist(x, self.c, p=self.distance_p_norm).argmin(1).cpu()
            )
        return torch.cat(results, dim=0).numpy()


if __name__ == "__main__":
    data, labels = load_fmnist("train")
    start_time = time.perf_counter()
    kmeans = MiniBatchKMeans(10, 1024, 20, device="cuda")
    pred_labels = kmeans.fit(data).predict(data)
    print("my train acc: ", unsupervised_clustering_accuracy(labels, pred_labels))
    print("duration: ", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    sk_kmeans = skminibatch(10, max_iter=20, random_state=42)
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
