import torch

empty_tensor = torch.empty(2, 2)
print(f"Empty tensor: {empty_tensor}")

random_tensor = torch.rand(2, 2)
print(f"Random tensor: {random_tensor}")

zeros_tensor = torch.zeros(2, 3)
print(f"Zeros tensor: {zeros_tensor}")

ones_tensor = torch.ones(2, 3)
print(f"Zeros tensor: {ones_tensor}")

# Printing Dtype
print(ones_tensor.dtype)

# Printing Size
print(empty_tensor.size())

ones_tensor_dtype = torch.ones(2, 5, dtype=torch.double)
print(f"Dtype specified: {ones_tensor_dtype}")
