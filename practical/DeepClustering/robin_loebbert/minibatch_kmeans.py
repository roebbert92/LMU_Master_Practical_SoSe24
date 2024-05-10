import torch
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler, TensorDataset
import numpy as np
from clustpy.data import load_fmnist
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy
from sklearn.cluster import MiniBatchKMeans as skminibatch
import time


class MiniBatchKMeans:
    k: int
    b: int
    t: int
    c: torch.Tensor
    rng: torch.Generator
    distance_p_norm: int
    optimizer: type[torch.optim.Optimizer]

    def __init__(
        self,
        k,
        mini_batch_size,
        iterations,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.9, "amsgrad": True},
        distance_p_norm=2,
        seed=42,
        device="cpu",
        max_no_improvement=30,
    ) -> None:
        self.k = k
        self.b = mini_batch_size
        self.t = iterations
        self.rng = torch.random.manual_seed(seed)
        self.device = device
        self.distance_p_norm = distance_p_norm
        self.optimizer = optimizer
        self.max_no_improvement = max_no_improvement
        self.optimizer_kwargs = optimizer_kwargs

    def fit(self, X: np.ndarray):
        dataset = TensorDataset(torch.tensor(X))
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
        self.c = torch.nn.Parameter(
            torch.tensor(X[random_idx], device=self.device), requires_grad=True
        )
        optimizer = self.optimizer([self.c], **self.optimizer_kwargs)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=0.9
        )
        n_samples = len(dataset)
        total_steps = (self.t * n_samples) // self.b
        steps_per_epoch = total_steps // self.t
        # epochs / iterations
        no_improvement = 0
        ewa_inertia_min = None
        ewa_inertia = None
        for i in range(1, total_steps):
            (x,) = next(iter(dataloader))
            # get random minibatch
            old_centers = self.c.detach().clone()
            optimizer.zero_grad()
            # foreach x in minibatch
            x = x.to(self.device)  # (b, d)
            # get nearest center index: dist = (b, k)
            dist = torch.cdist(x, self.c, p=self.distance_p_norm).argmin(1)  # (b, 1)
            one_hot = torch.nn.functional.one_hot(dist, num_classes=self.k).T  # (k, b)
            diff = one_hot.unsqueeze(2) * (
                x.unsqueeze(0) - self.c.unsqueeze(1)
            )  # (k, b, d)
            loss = torch.sum(diff**self.distance_p_norm)
            loss.backward()
            optimizer.step()
            if i % steps_per_epoch == 0:
                lr_scheduler.step()
            with torch.no_grad():
                batch_inertia = loss / x.size(0)
                if i > 1:
                    if ewa_inertia is None:
                        ewa_inertia = batch_inertia
                    else:
                        alpha = x.size(0) * 2.0 / (n_samples + 1)
                        alpha = min(alpha, 1)
                        ewa_inertia = ewa_inertia * (1 - alpha) + batch_inertia * alpha

                    if ewa_inertia_min is None or ewa_inertia < ewa_inertia_min:
                        no_improvement = 0
                        ewa_inertia_min = ewa_inertia
                    else:
                        no_improvement += 1

                print(
                    f"minibatch step {i}/{total_steps}: mean batch inertia {batch_inertia}, ewa inertia: {ewa_inertia}"
                )

                if torch.allclose(self.c, old_centers):
                    print("aborted, because clusters didn't change")
                    return self

                if no_improvement >= self.max_no_improvement:
                    print("aborted, because lack of improvement in inertia")
                    return self
        return self

    def predict(self, X: np.ndarray):
        # dataloader for one iteration of minibatches
        dataset = TensorDataset(torch.tensor(X))
        dataloader = DataLoader(dataset, self.b, shuffle=False)
        results = []
        with torch.no_grad():
            for batch in dataloader:
                (x,) = batch
                x = x.to(self.device)
                results.append(torch.cdist(x, self.c, p=self.distance_p_norm).argmin(1))
        return torch.cat(results, dim=0).cpu().numpy()


if __name__ == "__main__":
    data, labels = load_fmnist("train")
    start_time = time.perf_counter()
    kmeans = MiniBatchKMeans(10, 1024, 20, device="cuda")
    pred_labels = kmeans.fit(data).predict(data)
    print("my train acc: ", unsupervised_clustering_accuracy(labels, pred_labels))
    print("duration: ", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    sk_kmeans = skminibatch(
        10,
        init="random",
        batch_size=1024,
        max_iter=20,
        random_state=42,
        verbose=1,
        n_init=1,
        reassignment_ratio=0.0,
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
