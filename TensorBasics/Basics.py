import torch
import numpy as np

# Numpy arr
arr = np.array([1, 2, 3, 4, 5, 6])

# Creating tensor from numpy array
x = torch.from_numpy(arr)  # Specific for numpy array
x1 = torch.as_tensor(arr)  # Generalised

print(f"{type(x)}\n{type(x1)}")
print(f"Data type of the tensors: {x.dtype}")

arr_2d = np.arange(0, 25).reshape(5, -1)
print(arr_2d)

tensor_2d = torch.from_numpy(arr_2d)
print(tensor_2d)

"""The above methods which converts the array into tensor.. 
Doesn't make the copy of array. Both Array and Tensor shares same memory."""

new_array = np.arange(0, 50).reshape(10, -1)
print(new_array)
new_tensor = torch.tensor(new_array)
print(new_tensor)

# updating the array
new_array[0][0] = 123
print(new_array)
print(new_tensor)


# tocrch.Tensor automatically converts values to float
new_float_tensor = torch.Tensor(new_array)
print(new_float_tensor)

# Empty Tensor -> returns the block memory.. Basically used as placeholder.
empty_tensor = torch.empty(2, 2)
print(empty_tensor)


# Zeros
zeros_tensor = torch.zeros(4, 3, dtype=torch.int64)
print(zeros_tensor)

# ones
ones_tensor = torch.ones(4, 3)
print(ones_tensor)

# arange
print(torch.arange(0, 20, 2))
print(torch.linspace(0, 20, 15).reshape(5, -1))

# Create tensors from list, but have to define the data type.
print(torch.tensor([1, 2, 3, 4, 5]))
list_tensor = torch.tensor([1, 2, 3, 4, 5]).type(torch.int32)
print(list_tensor)

# random (uniform distribution)
print(torch.rand(4,3))

# random (normal distribution)
print(torch.randn(4,3))

# random int (high is excluded)
print(torch.randint(low=0, high=10, size=(5,5)))

# random like another tensor
ref_tensor = torch.zeros(2,5)
print(ref_tensor)
print(torch.rand_like(ref_tensor))
print(torch.randn_like(ref_tensor))
print(torch.randint_like(ref_tensor, low=20, high=30))

# Setting random seed
torch.manual_seed(42)
print(torch.randn(4,3))