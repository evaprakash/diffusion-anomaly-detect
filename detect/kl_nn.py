import torch
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

print("Loading data...")

batch_size = 500
dim = 512
X_train_len = 274515
X_test_len = 17326 
X_test_normal_len = 23465 

def concat_batches(X, X_leftover, dir_name):
    full_batches = X.shape[0]
    leftover_batch_size = X_leftover.shape[1]
    n = 0
    for file in os.listdir(dir_name):
        if (file == "diff_" + str(full_batches) + ".txt"):
            continue
        else:
            X[n] = np.load(dir_name + "/" + file)
            n += 1
    X_leftover[0] = np.load(dir_name + "/" + "diff_" + str(full_batches) + ".txt")
    X = np.reshape(X, (batch_size * full_batches, dim))
    X_leftover = np.reshape(X_leftover, (leftover_batch_size, dim))
    X = np.concatenate((X, X_leftover), axis=0)
    return X

X_train = concat_batches(np.zeros((int(X_train_len/batch_size), batch_size, dim)), np.zeros((1, X_train_len - int(X_train_len/batch_size) * batch_size, dim)), "full_train_resnet")
X_train = X_train.astype(np.float32)
X_test = concat_batches(np.zeros((int(X_test_len/batch_size), batch_size, dim)), np.zeros((1, X_test_len - int(X_test_len/batch_size) * batch_size, dim)), "full_test_anomaly_resnet")
X_test = X_test.astype(np.float32)
X_test_normal = concat_batches(np.zeros((int(X_test_normal_len/batch_size), batch_size, dim)), np.zeros((1, X_test_normal_len - int(X_test_normal_len/batch_size) * batch_size, dim)), "full_test_normal_resnet")
X_test_normal = X_test_normal.astype(np.float32)
print(X_train.shape, X_test.shape, X_test_normal.shape)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
X_test_normal = torch.from_numpy(X_test_normal)

y_train = np.ones((len(X_train),))
y_test = np.zeros((len(X_test),))
y_test_normal = np.ones((len(X_test_normal),))

x_size = 512
h_size = 512
y_size = 512

#train_loader = DataLoader(X_train, batch_size=5000, shuffle=True, num_workers=2, drop_last=True)
#test_loader = DataLoader(X_test, batch_size=5000, shuffle=True, num_workers=2, drop_last=True)
#test_normal_loader = DataLoader(X_test_normal, batch_size=5000, shuffle=True, num_workers=2, drop_last=True)

class OC_NN(nn.Module):
    def __init__(self):
        super(OC_NN, self).__init__()
        self.dense_out1 = nn.Linear(x_size, h_size)
        self.out2 = nn.Linear(h_size, y_size)

    def forward(self, x):
        z = self.dense_out1(x)
        y = self.out2(z)
        return y

def kl_div(x):
    mu = torch.mean(x, dim=1)
    std = torch.std(x, dim=1)
    p = torch.distributions.Normal(mu, std)
    q = torch.distributions.Normal(torch.zeros(mu.shape), torch.ones(std.shape))
    return torch.mean(torch.distributions.kl_divergence(p, q))

epochs = 10
learning_rate = 0.01

print("Building model...")
model = OC_NN()
model.train()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    print("Starting epoch ", str(epoch), "...")
    train_outputs = model(X_train)
    train_loss = kl_div(train_outputs)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    print("Train loss: ", train_loss.item())
    torch.save(model, 'oc_nn_model.pt')
    print("Done!")

model.eval()

print("Calculating final losses...")
train_outputs = model(X_train)
train_loss = kl_div(train_outputs)

test_outputs = model(X_test)
test_loss = kl_div(test_outputs)

test_normal_outputs = model(X_test_normal)
test_normal_loss = kl_div(test_normal_outputs)

print(train_loss, test_loss, test_normal_loss)

print("Done!")
