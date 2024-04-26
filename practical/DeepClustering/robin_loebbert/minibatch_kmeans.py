import torch
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler
import numpy as np


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

    def fit(self, X: Dataset):
        dataloader = DataLoader(X, self.b, shuffle=True, generator=self.rng)
        # init C with k random entries from X
        self.c = torch.tensor(
            Subset(
                X,
                torch.randint(
                    high=len(X),
                    size=self.k,
                    device="cpu",
                    generator=self.rng,
                    requires_grad=False,
                ),
            ),
            device=self.device,
        )
        # zero init per center counts
        v = torch.zeros(self.c.size(), device=self.device)
        # epochs / iterations
        for _ in range(self.t):
            # get random minibatch
            for batch in dataloader:
                # foreach x in minibatch
                # get nearest center index: dist = (b, k)
                dist = torch.cdist(batch, self.c, p=self.distance_p_norm)
                d = dist.argmax(2)
                # foreach x in minibatch --> loop needed? or is sum of steps enough?
                # maybe
                # update the center counts for this center
                v_indices, counts = d.unique(sorted=True, return_counts=True, dim=0)
                v = v.index_add_(0, v_indices, counts)
                # get per center learning rate
                eta = 1 / v
                # one hot of batch : x = (b, k, d)
                x = torch.zeros_like(dist).scatter_(2, d.unsqueeze(2), 1.0) * batch
                # take gradient step: self.c = (k,d), eta = (k,), batch = (b,d), d = (b,)
                self.c = (1 - eta) * self.c + eta * torch.index_fill()

                pass
            pass
        pass

    def predict(self, X: Dataset):
        # dataloader for one iteration of minibatches
        # forearch x in minibatch:
        # get nearest center index
        pass
