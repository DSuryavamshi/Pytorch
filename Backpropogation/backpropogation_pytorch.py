# This the code for backpropogation.png

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
# f(x) = y = w * x
y_hat = w * x
loss = (y - y_hat)**2

print(loss)

loss.backward()
print(w.grad)

# update weights
# more iterations