import torch

x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(f"X : {x}")
print(f"Y : {y}")

# Addition
z_add = x + y  # Element wise addition
print(f"Element wise addition: {z_add}")
z_add = torch.add(x, y)  # Same as Element wise addition
print(f"Element wise addition inbuilt method: {z_add}")
y.add_(x)  # Inplace addition. (every method in pytorch with trailing '-' does inplace operation)
print(f"Inplace addition: {y}")

# Same for subtraction, division, multiplication

# Slicing is similar to numpy
print(x[0, :])  # Printing first row with all the columns

# Reshaping is done using .view. It's similar to numpy
a = torch.rand(4,5)
print(f"Actual tensor:\n{a}")
print(f"Reshaped tensor:\n {a.view(10,-1)}")