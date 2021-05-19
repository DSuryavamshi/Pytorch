import torch
import torch.nn as nn


x = torch.tensor([2.0, 1.0, 0.1])
output = torch.softmax(x, dim=0)
print(f"Softmax output for x = {x}:\n{output}")

# While calculating CrossEntropyLoss using pytorch remember,
# 1. softmax is implemented automatically (nn.LogSoftmax, nn.NLLLoss)
# 2. One Hot Encoded not is needed, Y should have correct class label
# 3. Predicted outcome has raw scores, not softmax

loss = nn.CrossEntropyLoss()
y = torch.tensor([0])
# size of y_pred = n_samples (1) x n_classes (3)
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.5]])
loss_good = loss(y_pred_good, y)
loss_bad = loss(y_pred_bad, y)
print(f"Loss for good prediction: {loss_good}")
print(f"Loss for bad prediction: {loss_bad}")

good_value, good_class = torch.max(y_pred_good, dim=1)
bad_value, bad_class = torch.max(y_pred_bad, dim=1)

print(good_class.item())
print(bad_class.item())

# multiple samples
# 3 samples
# 3 classes
y_classes = torch.tensor([0, 1, 2])
y_pred_good = torch.tensor([[2.0, 1.0, 0.1], [2.0, 2.8, 0.1], [2.0, 1.0, 5.0]])
y_pred_bad = torch.tensor([[2.0, 10.0, 0.1], [5.0, 2.8, 0.1], [20.0, 1.0, 5.0]])

mul_loss_good = loss(y_pred_good, y_classes)
mul_loss_bad = loss(y_pred_bad, y_classes)
print(f"Loss for good prediction: {loss_good}")
print(f"Loss for bad prediction: {loss_bad}")

good_value, good_class = torch.max(y_pred_good, dim=1)
bad_value, bad_class = torch.max(y_pred_bad, dim=1)
print(good_class)
print(bad_class)
