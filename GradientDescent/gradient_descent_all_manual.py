import numpy as np

# f(x) = y = w * x

x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10], dtype=np.float32)
w = 0.0

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# gradient
# MSE = 1/N * (w*x  - y) ** 2
# dJ/dw = 1/2 * 2*x * (wx - y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()


print(f"prediction before training, f(5) = {forward(5):.3f}")

# training

learning_rate = 0.01
ephocs = 10

for epoch in range(ephocs):
    y_pred = forward(x)
    loss_val = loss(y, y_pred)
    dw = gradient(x=x, y=y, y_predicted=y_pred)

    # update weights
    w -= learning_rate * dw

    print(f"epoch:{epoch+1} w:{dw:.3f} loss:{loss_val:.3f}")

print(f"prediction before training, f(5) = {forward(5):.3f}")
