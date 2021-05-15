import numpy as np
import torch

# f(x) = y = w * x

x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)
w = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# gradient
# MSE = 1/N * (w*x  - y) ** 2
# dJ/dw = 1/2 * 2*x * (wx - y)
# def gradient(x, y, y_predicted):
#     return np.dot(2*x, y_predicted - y).mean()


print(f"prediction before training, f(5) = {forward(5):.3f}")

# training

learning_rate = 0.01
ephocs = 100

for epoch in range(ephocs):
    y_pred = forward(x)
    loss_val = loss(y, y_pred)
    # dw = gradient(x=x, y=y, y_predicted=y_pred)
    # backward pass
    loss_val.backward()  # dl/dw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # making grads zero before next iter
    if epoch % 10 == 0:
        print(f"epoch:{epoch+1} w:{w.grad:.3f} loss:{loss_val:.3f}")

    w.grad.zero_()

print(f"prediction before training, f(5) = {forward(5):.3f}")
