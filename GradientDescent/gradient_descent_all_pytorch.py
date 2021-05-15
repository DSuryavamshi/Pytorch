# General Training pipelne
# 1) Design model - input_size, output_size, forward pass
# 2) Construct loss and Optimizer
# 3) Training Loop
#       - forward pass: compute  prediction
#       - backward pass: gradients
#       - update weights


import torch.nn as nn
import torch

# f(x) = y = w * x

x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8], [10]], dtype=torch.float32)
w = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)


n_samples, n_features = x.shape
model = nn.Linear(in_features=n_features, out_features=n_features)
x_test = torch.tensor([5], dtype=torch.float32)
print(f"prediction before training, f(5) = {model(x_test).item():.3f}")

# training

learning_rate = 0.01
ephocs = 200

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(ephocs):
    y_pred = model(x)

    # backward pass
    loss = loss_fn(y, y_pred) 
    loss.backward()  # dl/dw

    # update weights
    optimizer.step()

    # making grads zero before next iter
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch:{epoch+1} w:{w[0][0]:.3f} loss:{loss:.3f}")

    optimizer.zero_grad()

print(f"prediction after training, f(5) = {model(x_test).item():.3f}")
