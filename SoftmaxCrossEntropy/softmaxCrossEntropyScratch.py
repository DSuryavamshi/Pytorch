import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy_loss(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(f"Softmax output for x = {x}:\n{outputs}")

# for cross_entropy, y must be one-hot encoded
# Assume three classes A, B, C
# for class A : y = [1, 0, 0]
# for class B : y = [0, 1, 0]
# for class C : y = [0, 0, 1]
y = np.array([1, 0, 0])

y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
loss_good = cross_entropy_loss(y, y_pred_good)
loss_bad = cross_entropy_loss(y, y_pred_bad)
print(f"Loss for good prediction: {loss_good}")
print(f"Loss for bad prediction: {loss_bad}")
