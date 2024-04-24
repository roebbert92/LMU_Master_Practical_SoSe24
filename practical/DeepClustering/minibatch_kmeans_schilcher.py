import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt

# TODO:
# (with autograd)

class MinibatchKmeans:
    """
    Implementation of minibatch-kmeans for a given number of clusters.
    """

    def __init__(self, k:int):
        """

        Args:
            k      (int): number of clusters

        """
        self.k = k # number of clusters
        self.iter = 1 # can be used for analysis
        self.X = None # data matrix
        self.centers = None # representatives of the cluster
        self.init_centers = None # can be used for analysis

    def _get_minibatch(self, batch_size) -> torch.Tensor:
        """
        Returns a minibatch of the stored data matrix X

        Args:
            batch_size  (int): the number of samples in the minibatch

        Returns:
                      (Tensor): the minibatch of given size
        """
        if not isinstance(batch_size, int):
            raise TypeError("batch_size should be of type integer")
        if batch_size < 1:
            raise ValueError("Choose a batch size bigger than 0!")
        if X is None:
            raise ValueError("data matrix X is None. Make sure you call fit() before")
        indices = torch.randperm(len(self.X))[:batch_size]
        return self.X[indices]

    def _get_assignments(self,M):
        """
        Computes the assignments of the samples in the given data matrix M (<number of samples> x <number of features>)
        to the clusters represented by the cluster centers.

        Args:
            M   (Tensor): data tensor of dimension (<number of samples> x <number of features>)

        Returns:
                (Tensor): predicted class labels as a one dimensional tensor
        """
        if M is None:
            raise ValueError("Given data matrix cannot be None")
        if self.centers is None:
            raise ValueError("No centers available. Make sure you call fit() first")
        if not isinstance(M, torch.Tensor):
            raise TypeError("M must be a pytorch Tensor")
        
        distance_matrix = torch.cdist(M,self.centers,p=2) # kmeans uses L_2 norm (euclidean distance)
        assignments = torch.argmin(distance_matrix.squeeze(), dim=1)
        return assignments


    def fit(self, X: np.ndarray, batch_size: int, max_iterations=100) -> np.ndarray:
        """
        Performs the Minibatch-Kmeans clustering on the given dataset X. 

        Args:
            X              (array): The data array of shape (<number of samples> x <number of features>).
            batch_size       (int): The number of samples in the minibatch
            max_iterations   (int): The number of maximal iterations which should be performed

        Returns:
            out            (array): The predicted class lables for each sample as a 1 dimensional numpy array
            """
        if X is None:
            raise ValueError("Given data matrix X is None")
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError("Data matrix X should be a 2 dimensional numpy array")
        if not isinstance(batch_size, int):
            raise TypeError("batch_size should be of type integer")
        if not isinstance(max_iterations, int):
            raise TypeError("max_iterations should be of type integer")
        if batch_size < 1:
            raise ValueError("Choose a batch size bigger than 0!")
        if max_iterations < 1:
            raise ValueError("Choose a maximum number of iterations bigger than 0!")
    
        self.X = torch.from_numpy(X)
        self.batch_size = batch_size
        # initialise centers with k randomly picked samples from X
        self.centers = self._get_minibatch(self.k) 
        self.init_centers = self.centers.detach().clone() # store initial centers seperatly for later analysis
        
        # initialise center counts used for adopting learnig rate
        center_counts = torch.zeros(self.k, dtype=torch.int)

        self.iter = 1
        # store centers of last iteration to check if centers are stable
        old_centers = torch.zeros(self.centers.shape, dtype=self.centers.dtype)
        
        while not torch.allclose(self.centers, old_centers, rtol=0.01) and self.iter < max_iterations:

            old_centers = self.centers.detach().clone()

            M = self._get_minibatch(batch_size)
            # assign each sample in minibatch to its nearest center
            assignment = self._get_assignments(M)
            
            # iteration over each sample is necessary in order to be able to adopt learning rate as stated in paper
            for i, sample in enumerate(M):
                # label of current sample 
                label = assignment[i]
                # adopt learning rate
                center_counts[label] += 1
                learning_rate = 1/center_counts[label]  
                # make gradient step
                self.centers[label] = self.centers[label] + learning_rate*(sample - self.centers[label])
            self.iter +=1
        return self._get_assignments(self.X).numpy()

# Test the implementation
k = 4
X, y_true = make_blobs(n_samples=300, centers=k, cluster_std=0.70, random_state=0)
kmeans = MinibatchKmeans(k)
labels_pred = kmeans.fit(X, 50)
plt.scatter(X[:,0],X[:,1], c=labels_pred)
plt.scatter(kmeans.centers[:,0], kmeans.centers[:,1], marker="X", c="red", label="Final centers")
plt.scatter(kmeans.init_centers[:,0], kmeans.init_centers[:,1], c="darkgrey", label="Initial centers")
plt.legend()
plt.title(f"Minibatch-kmeans: {kmeans.iter} iterations needed, NMI={normalized_mutual_info_score(y_true, labels_pred)}")
plt.show()

