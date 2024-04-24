from minibatch_kmeans_schilcher import MinibatchKmeans
import unittest
import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score

class TestMinibatchKmeans(unittest.TestCase):

    def test_get_minibatch(self):
        kmeans = MinibatchKmeans(3)
        kmeans.X = torch.rand((5,4))
        # generate a minibatch of size 2 out of X
        mb = kmeans._get_minibatch(batch_size=2)
        self.assertEqual(mb.shape, (2,4))
        self.assertTrue(torch.isin(mb,kmeans.X).all().item())

    def test_get_assignments(self):
        kmeans = MinibatchKmeans(4)
        kmeans.X = torch.tensor([[0,-5], [20, 15], [38, 2], [190, -100]], dtype=torch.float)
        kmeans.centers = torch.tensor([[200, -90], [40, 0], [22, 23], [-10, -3]], dtype=torch.float)
        labels = kmeans._get_assignments(kmeans.X)
        # ckeck if assignments are done right
        self.assertTrue(torch.eq(labels, torch.tensor([3,2,1,0])).all().item())

    def test_fit_1(self):
        X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.20, random_state=0)
        kmeans = MinibatchKmeans(4)
        y_pred = kmeans.fit(X, 50)
        # check if each sample got a labe
        self.assertEqual(len(y_pred), len(X))
        # check if really 4 classes were outputed
        self.assertTrue(np.equal(np.unique(y_pred), np.array([0,1,2,3])).all().item())


    def test_fit_2(self):
        X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.20, random_state=0)
        kmeans = MinibatchKmeans(4)
        # mutual information should have high value averaged over multiple runs (depending on quality of random initialisation)
        nmi = 0
        for i in range(50):
            y_pred = kmeans.fit(X, 50)
            nmi += normalized_mutual_info_score(y_true, y_pred)
        self.assertTrue(nmi/50 > 0.8)


if __name__ == '__main__':
    unittest.main()