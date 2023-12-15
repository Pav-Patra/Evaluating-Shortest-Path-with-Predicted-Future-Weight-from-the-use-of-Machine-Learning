import torch
import numpy as np

# Tensor Initialised Directly from Data
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

# Tensor Initialised from a NumPy Array
np_array =  np.array(data)
x_np = torch.from_numpy(np_array)

# Tensor Initialised from another tensor
x_ones = torch.ones_like(x_data)   # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n" )

x_rand = torch.rand_like(x_data, dtype=torch.float)   # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


# Tensor initialised with random or constant values (shape is a tuple of tensor dimensions - determines the dimensionality of the output tensor)
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}\n")

# Tensor Attributes - these describe a tensor's shape, datatype, and the device on which they are stored
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}\n")

# Tensor Operations
# we move our tensor to the GPU if available
print("CUDA GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: {tensor.device}\n")

# slice on tensor where all [1] (second) column values are 0
tensor = torch.ones(4,4)
tensor[:,1] = 0
print(tensor)

print("\n")

# Joining Tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

print("\n")

# Multiplying Tensors

# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax
print(f"tensor * tensor \n {tensor * tensor} \n")

# This computes the matrtix multiplication between two tensors
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternatice syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T} \n")

# In-Place Operations
print(tensor, "\n")
tensor.add_(5)
print(tensor, "\n")

# Tebsor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}\n")

# a change to tensor reflects in the NumPy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}\n")

# Numpy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

# changes to the numpy array reflect in the tensor
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}\n")
