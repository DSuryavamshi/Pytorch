import torch
import numpy as np

x = torch.arange(6).reshape(3, 2)
print(x)
print(x[1, 1])  # even after indexing tensor is returned
print(x[:, 1])  # indexing for a column but it'll return in form of a row.
print(x[:, 1:])  # if we slice the tensor, we get it in column format
#  view v/s reshape link
#  (https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch/49644300#49644300)

y_view = x.view(2, -1)
print(y_view)
y_reshape = x.reshape(2, -1)
print(y_reshape)
# X remains unchanged
print(x)
x[0][0] = 9999
print(x)
print(y_reshape)
print(y_view)

a = torch.Tensor([1, 2, 3])
b = torch.Tensor([4, 5, 6])
# Addition
print(a + b)
print(a.add(b))  # a.add_(b) method will reassign the sum to a
# Multiplication
print(a * b)
print(a.mul(b))
# division
print(a / b)
print(a.div(b))
# Modulus
print(a % b)
# subtraction
print(a - b)
print(a.sub(b))

# dot product
print(a.dot(b))

# Matirx Mul
a_m = torch.arange(10).reshape(5, -1)
b_m = torch.arange(10).reshape(-1, 5)
print(a_m)
print(b_m)
print(torch.mm(a_m, b_m))
print(a_m @ b_m)

# Euclidian Norm
en = torch.Tensor(np.arange(10).reshape(2, -1))
print(en.norm())

# Number of elements
print(en.numel())
print(len(en))  # gives number of rows for multi dimension vectors
