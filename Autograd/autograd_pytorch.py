import torch

# If you want to use this vairable for differentiation in future, we need to keep required_grad as True
x = torch.randn(2, requires_grad=True)
y = x + 2  # This equation is using X.
print(y)
z = y * y * 2
# print(z)
z = z.mean()
print(z)
# Gradients can be implicitly calculated only for scalar values.
z.backward()
print(f"gradients for scalar: {x.grad}")
# to calculate gradients for non scalar values, we need to pass a vector. As pytorch internally follows Jacobian Matirx
v = torch.randn_like(y)
y.backward(v)
print(f"gradients for non scalar: {x.grad}")

# While training, we need to make sure our gradients are not being tracked by pytorch. We can do it following ways
x.requires_grad_(False)
print(f"After .requires_grad_():{x}")
m = x.detach()
print(f"After .detach():{m}")

a = torch.randn(3, requires_grad=True)
with torch.no_grad():
    y = a + 2
    print(f"Printing in with no_grad():\n{y}")

# Every time we call .backwards, the gradients are summed up and stored in .grad, We need to avoid this.
# We can use .grad.zero_() function to set the gradients to zero.
