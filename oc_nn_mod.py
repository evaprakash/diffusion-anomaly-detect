import os
import numpy as np
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader

from torchmetrics import AUROC

import cae

from itertools import zip_longest
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

print("Loading data...")

X_train = np.zeros((5842, 512))
i = 0
for file in os.listdir("/vision/u/eprakash/diffae/train_viz_results"):
    X_train[i] = np.load("/vision/u/eprakash/diffae/train_viz_results/" + file)
    i += 1
print("Loaded train!")
X_train = torch.from_numpy(X_train.astype(np.float32))


X_test = np.zeros((1000, 512))
i = 0
for file in os.listdir("/vision/u/eprakash/diffae/mnist_anomaly_conds/"):
    X_test[i] = np.load("/vision/u/eprakash/diffae/mnist_anomaly_conds/" + file)
    i += 1
print("Loaded test!")
X_test = torch.from_numpy(X_test.astype(np.float32))

X_test_normal = np.zeros((1000, 512))
i = 0
for file in os.listdir("/vision/u/eprakash/diffae/mnist_normal_conds/"):
    X_test_normal[i] = np.load("/vision/u/eprakash/diffae/mnist_normal_conds/" + file)
    i += 1
print("Loaded test normal!")
X_test_normal = torch.from_numpy(X_test_normal.astype(np.float32))

train_loader = DataLoader(X_train, batch_size=384, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(X_test, batch_size=384, shuffle=True, num_workers=2, drop_last=True)
test_normal_loader = DataLoader(X_test_normal, batch_size=384, shuffle=True, num_workers=2, drop_last=True)

dataset_size = len(X_train)
x_size = 512
h_size = 128
y_size = 1

class OC_NN(nn.Module):
    def __init__(self):
        super(OC_NN, self).__init__()
        self.dense_out1 = nn.Linear(x_size, h_size)
        #self.act = nn.Sigmoid()
        self.out2 = nn.Linear(h_size, y_size)

    def forward(self, img):
        #w1 = self.act(self.dense_out1(img))
        a = self.dense_out1(img)
        b = self.out2(a)
        V = self.dense_out1.weight
        w = self.out2.weight
        return a, V, w

model = OC_NN()
model.load_state_dict(torch.load("best/4/nn_model.pt"))
#model.to(device)
nu = 0.15

def nnscore(a, w):
    return torch.matmul(a, w.transpose(1,0))

def ocnn_loss(nu, r, a, V, w, N):
    term1 = 0.5 * torch.sum(V**2)
    term2 = 0.5 * torch.sum(w**2)
    term3 = 1/N * 1/nu * torch.mean(F.relu(r - nnscore(a, w)))
    term4 = -r

    return term1 + term2 + term3 + term4

num_epochs = 0

lr = 1e-2

optimizer = optim.SGD(model.parameters(), lr=0.0001)

best_loss = 10000

N = 384

r_value =  0.013907559122890234#0.011273916438221931 #np.random.normal(0, 1) #0.0868869110941887
r_value_original = r_value
print("R Value: ", str(r_value_original))
r = torch.from_numpy(np.full((N, y_size), r_value))

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    model.train()  # Set model to training mode
    running_loss = 0.0

    # Iterate over data.
    for inputs in train_loader:
        #inputs.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        # track history if only in train
        a, V, w = model(inputs)
        #print(w1.shape, w2.shape, V.shape, r.shape)
        loss = ocnn_loss(nu, r, a, V, w, N)
        loss = loss.mean()
        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()
        
        r = r.cpu().detach().numpy()
        r_value = np.quantile(r, q=nu)
        r = torch.from_numpy(np.full((N, y_size), r_value))
        
        # statistics
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / dataset_size
    
    print('Loss: {:.4f} '.format(epoch_loss))
    print('Epoch = %d, r = %f'%(epoch+1, r_value))

    # deep copy the model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), 'nn_model.pt')
        a, V, w = model(inputs)
        r = nnscore(a, w)
        r = r.cpu().detach().numpy()
        r_value = np.quantile(r, q=nu)
        r = torch.from_numpy(np.full((N, y_size), r_value))
        with open("r_value.txt", "w") as fp:
            fp.write("Original: " + str(r_value_original) + "\nBest: " + str(r_value) + "\n")
    print()

print("Calculating evaluation metrics...")
model.eval()
with torch.no_grad():
    i = 0
    train_score = np.ones((dataset_size,))
    train_loader = DataLoader(X_train, batch_size=N, shuffle=True, num_workers=2, drop_last=True)
    for inputs in train_loader:
        a, V, w = model(inputs)
        score = nnscore(a, w)
        score = score.detach().numpy().flatten() - r.detach().numpy().flatten()
        if (i == int(dataset_size/N)):
            train_score[i * N : ] = score
        else:
            train_score[i * N : (i + 1) * N] = score
        i += 1
    print("Classified anomaly: ", str(np.sum(train_score<0)/dataset_size))
    print("Classified normal: ", str(np.sum(train_score>=0)/dataset_size))

    i = 0
    test_score = np.zeros((len(X_test),))
    for inputs in test_loader:
        a, V, w = model(inputs)
        score = nnscore(a, w)
        score = score.detach().numpy().flatten() - r.detach().numpy().flatten()
        if (i == int(len(X_test)/N)):
            test_score[i * N : ] = score
        else:
            test_score[i * N : (i + 1) * N] = score
        i += 1
    print("Classified anomaly: ", str(np.sum(test_score<0)/len(X_test)))
    print("Classified normal: ", str(np.sum(test_score>=0)/len(X_test)))

    i = 0
    test_normal_score = np.zeros((len(X_test_normal),))
    for inputs in test_normal_loader:
        a, V, w = model(inputs)
        score = nnscore(a, w)
        score = score.detach().numpy().flatten() - r.detach().numpy().flatten()
        if (i == int(len(X_test_normal)/N)):
            test_normal_score[i * N : ] = score
        else:
            test_normal_score[i * N : (i + 1) * N] = score
        i += 1
    print("Classified anomaly: ", str(np.sum(test_normal_score<0)/len(X_test_normal)))
    print("Classified normal: ", str(np.sum(test_normal_score>=0)/len(X_test_normal)))

    y_test_score = np.concatenate((test_score, test_normal_score))

    y_score = np.concatenate((train_score, test_score, test_normal_score))

    y_train = np.ones((len(train_score),))
    y_test = np.zeros((len(test_score),))
    y_test_normal = np.ones((len(test_normal_score),))

    y_test_true = np.concatenate((y_test, y_test_normal))
    
    y_true = np.concatenate((y_train, y_test, y_test_normal))

    average_precision_test = average_precision_score(y_test_true, y_test_score)

    print('Average precision-recall score test: {0:0.4f}'.format(average_precision_test))

    roc_score_test = roc_auc_score(y_test_true, y_test_score)

    print('ROC score test: {0:0.4f}'.format(roc_score_test))
    
    average_precision = average_precision_score(y_true, y_score)

    print('Average precision-recall score: {0:0.4f}'.format(average_precision))

    roc_score = roc_auc_score(y_true, y_score)

    print('ROC score: {0:0.4f}'.format(roc_score))
