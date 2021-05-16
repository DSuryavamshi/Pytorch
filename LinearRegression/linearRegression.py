import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Prepare data
x_np, y_np = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=1)

x_tensor = torch.from_numpy(x_np.astype(np.float32))
y_tensor = torch.from_numpy(y_np.astype(np.float32))
y_tensor = y_tensor.view(y_tensor.shape[0], 1)
inp_features = x_tensor.shape[1]
out_features = y_tensor.shape[1]

# Model
model = nn.Linear(in_features=inp_features, out_features=out_features)

# loss
loss = nn.MSELoss()

# optimizer
optim = torch.optim.SGD(params=model.parameters(), lr=0.01)

# training loop
epochs = 1000

plt.scatter(x_np, y_np)
for epoch in range(epochs):
    # forward pass
    y_pred = model(x_tensor)

    # loss
    l = loss(y_pred, y_tensor)

    # backward pass
    l.backward()

    # update weights
    optim.step()

    # setting grads to zero for next iter
    optim.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch:{epoch+1} loss:{l:.3f}")
        # plot
        predicted = model(x_tensor).detach().numpy()  # removing requires_grad
        plt.plot(x_np, predicted, label=f'epoch:{epoch+1}')


plt.legend()
plt.grid()
plt.show()
