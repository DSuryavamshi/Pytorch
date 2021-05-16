from scipy.sparse.construct import rand, random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.nn.modules import linear
from torch.nn.modules.activation import Sigmoid

# PrepareData
data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# model

class LogisticRegression(nn.Module):
    def __init__(self, inp_features, out_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=inp_features, out_features=out_features)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression(inp_features=X_train.shape[1], out_features=y_train.shape[1])

# loss
loss = nn.BCELoss()

# optimizer
optim = torch.optim.SGD(params=model.parameters(), lr=0.01)

# training loop:
epochs = 10000

for epoch in range(epochs):
    # foward pass
    y_pred = model.forward(X_train)

    # loss
    l = loss(y_pred, y_train)

    # backward pass
    l.backward()

    # update weights
    optim.step()

    # clear grads
    optim.zero_grad()

    if epoch % 1000 == 0:
        print(f"Epoch:{epoch+1} loss:{l:.3f}")

#Evaluation
with torch.no_grad():
    y_test_pred = model.forward(X_test).round()

    accuracy = y_test_pred.eq(y_test).sum()/y_test.shape[0]
    print(f"Accuracy of the model: {accuracy:.3f}")