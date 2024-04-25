import torch.utils
from practical.DeepClustering import minibatchkmeans
import torch
import numpy as np

# Fixed values
N_CLUSTERS = 2
MAX_ITER = 20
BATCH_SIZE = 5
RANDOM_STATE = 0
INIT_METHOD = 'random'
TESTDATA_ROWS = 100
TESTDATA_COLS = 3



np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Test dataset
data = np.random.rand(TESTDATA_ROWS, TESTDATA_COLS)
torch_data = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

def test_instantiate_miniBatchKMeans_object():
    # Initialize MiniBatch k-Means
    mbkm = minibatchkmeans.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
    
    assert isinstance(mbkm, minibatchkmeans.MiniBatchKMeans)
    assert mbkm.n_clusters == N_CLUSTERS
    assert mbkm.max_iter == MAX_ITER
    assert mbkm.batch_size == BATCH_SIZE
    assert mbkm.labels_ == None
    assert mbkm.cluster_centers_ == None
    assert mbkm.random_state == RANDOM_STATE

def test_randomly_select_centers():
    # Test data to test the random selection result on.
    test_data = data[np.random.choice(data.shape[0], N_CLUSTERS, replace=False)]
    
    # Assert for numpy array
    mbkm_class = minibatchkmeans.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
    assert np.all(test_data == np.array(mbkm_class._random_centroids(data).tolist()))
    
    # Assert for torch object
    mbkm_class = minibatchkmeans.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
    assert torch.all(torch.tensor(test_data) == mbkm_class._random_centroids(torch_data))
    
def test_minibatchkmeans_algorithm_result_size():
    # Test if the output is of correct size
    
    # Init object
    mbkm = minibatchkmeans.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
    cluster_centers = mbkm._random_centroids(torch_data)
    
    # Assert if the result is of the expected size
    assert torch.Size([N_CLUSTERS, TESTDATA_COLS]) == cluster_centers.shape
    
def test_assigning_labels():
    # Initialize a new MiniBatch k-Means Object
    mbkm = minibatchkmeans.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
    
    # Retrieve the initial random cluster centers
    cluster_centers_initial = mbkm._random_centroids(torch_data)
    
    # Store cluster centers in the object data
    mbkm.cluster_centers_ = cluster_centers_initial
    
    # Retrieve the initial dataset labels
    labels_initial = mbkm._assign_labels(torch_data)
    
    # Assert that they are of correct length and have the correct range of 0..N_CLUSTERS
    assert labels_initial.shape[0] == TESTDATA_ROWS # Length/Amount
    assert torch.all(labels_initial.unique() == torch.tensor([i for i in range(0, N_CLUSTERS)])) # 0..N_CLUSTERS
    
    
    
    
def test_minibatchkmeans_full_algorithm():
    # Init object
    mbkm = minibatchkmeans.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
    mbkm.fit(torch_data)
    
    # Assert if final label result is again of correct size and correct amount of clusters (0..N_CLUSTERS)
    assert mbkm.labels_.shape[0] == TESTDATA_ROWS # Length/Amount
    assert torch.all(mbkm.labels_.unique() == torch.tensor([i for i in range(0, N_CLUSTERS)])) # 0..N_CLUSTERS
    