import torch
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import models
from model.resnet import *
from torchvision.utils import save_image
from PIL import Image

x_size = 512
h_size = 512
y_size = 512
emb_size = 512

class KL_NN(nn.Module):
    def __init__(self):
        super(KL_NN, self).__init__()
        self.dense_out1 = nn.Linear(x_size, h_size)
        self.out2 = nn.Linear(h_size, y_size)
        self.resnet18 = ResNetEncoderModel(emb_size) 
        '''
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)
        for p in resnet18.parameters():
            p.requires_grad = False
        self.transform = T.Resize(size = (224))
        '''
    def forward(self, output, target):
        output = output.float()
        target = target.float()
        save_results(output, "output")
        save_results(target, "target")
        output_mod = self.resnet18(output).view(output.shape[0], emb_size)
        target_mod = self.resnet18(target).view(target.shape[0], emb_size)
        #output_mod = self.resnet18(self.transform(output)).view(output.shape[0], emb_size)
        #target_mod = self.resnet18(self.transform(target)).view(target.shape[0], emb_size)
        diff = torch.abs(output_mod - target_mod)
        #with open("kl_diff_normal.txt", "wb") as fp:
        #    np.save(fp, diff.clone().cpu().detach().numpy())
        z = self.dense_out1(diff)
        y = self.out2(z)
        return kl_div(y)

def save_results(frame_batch, name):
    frame_batch = frame_batch.permute(0, 2, 1, 3, 4)
    for n in range(frame_batch.shape[0]):
        for i in range(frame_batch.shape[1]):
            img = frame_batch[n][i].permute(1, 2, 0).cpu().detach().numpy()
            img = Image.fromarray(img, mode="RGB")
            img_name = "save_results_" + name + "/img_" + str(i) + ".png"
            img.save(img_name)

def kl_div(x):
    device = x.get_device()
    print(x.shape)
    mu = torch.mean(x, dim=1)
    std = torch.std(x, dim=1)
    print("mu:", mu)
    print("std:", std)
    p = torch.distributions.Normal(mu, std)
    q = torch.distributions.Normal(torch.zeros(mu.shape).to(device), torch.ones(std.shape).to(device))
    return torch.distributions.kl_divergence(p, q)
