import torch
import numpy as np

# Numpy to tensor
a = np.array([1, 2, 3, 4, 5])
b = torch.from_numpy(a)
print(type(b))
print(type(a))

# Tensor to numpy
x = torch.tensor([5, 6, 7, 8, 9])
y = x.numpy()
print(type(x))
print(type(y))

# Note: If both tensor and numpy array are in CPU, both of them share same memory, meaning if you change one of them
# the other will get changed too

# Creating a  Tensor in GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_tensor = torch.ones(2, 3, device=device)
    gpu_tensor1 = torch.rand(2, 3, device=device)
    print(gpu_tensor1 + gpu_tensor)
    # When the tensor is in GPU, we cannot convert it to numpy array as numpy doesn't support GPU operations.
    z = gpu_tensor.to('cpu')
    print(z.numpy())