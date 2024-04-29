from practical.DeepClustering import MiniBatchKmeans
import torch

# generate the input, 50 data poing with 10 features for each data point
input  = torch.randn(50,10)
n_clusters = 5
batch_size = 10
n_iter = 10
model = MiniBatchKmeans(n_clusters,batch_size,n_iter)
centers = model.forward(input)
print(centers)

# output :tensor([[ 0.0802, -0.3703,  0.3922, -1.2998,  0.5223, -0.4484,  0.0399,  0.1811,
#          -0.4285,  0.4424],
#         [-0.8336,  1.0182,  0.4681,  0.1337,  0.0667,  0.3909, -0.1763,  0.2117,
#          -1.7768, -0.6515],
#         [ 0.1637,  0.0533,  0.1354, -0.2415,  0.9188, -0.0394,  1.0039, -0.8679,
#           1.4603,  0.2197],
#         [-0.0606,  0.2250, -0.2642,  1.0825,  0.7565, -1.5832, -1.6241, -1.0770,
#          -0.3406, -1.6672],
#         [ 0.4951, -0.3880,  0.2946,  1.2581,  0.1129, -0.0109, -0.5472,  0.1261,
#           0.0413, -0.1444]])
