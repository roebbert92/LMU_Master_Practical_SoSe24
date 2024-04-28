import torch
class MiniBatchKmeans:
    def __init__(self, n_clusters,batch_size,n_iter):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.n_iter = n_iter
    def forward(self, input):
        samples, features = input.shape
        # initialize centroid for each cluster
        n_centroids = input[torch.randint(0,samples,(self.n_clusters,))] 
        # v represent the number of data point for each cluster
        v = torch.zeros(self.n_clusters)
        for i in range(self.n_iter):
            # randomly select batch_size indices for creating the current batch
            batch = torch.randint(0, samples, (self.batch_size,))
            input_batch = input[batch]
            # compute distances from batch samples to centroids
            d = torch.cdist(input_batch, n_centroids)
            # find the nearest centroid for each sample in the batch
            near_centroid = torch.argmin(d,dim=1)
            # update the centroids based on samples
            for id, x in enumerate(input_batch):    
                c = near_centroid[id]
                v[c] +=1
                lr = 1/v[c]
                n_centroids[c] = (1-lr)*n_centroids[c]+lr*x
        return n_centroids
