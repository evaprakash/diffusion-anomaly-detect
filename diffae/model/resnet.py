import math
from dataclasses import dataclass
from numbers import Number
from typing import NamedTuple, Tuple, Union
from torchvision import models
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from choices import *
from config_base import BaseConfig
from .blocks import *

from .nn import (conv_nd, linear, normalization, timestep_embedding,
                         torch_checkpoint, zero_module)

class ResNetEncoderModel(nn.Module):
    def __init__(self, enc_out_channels):
        super().__init__()
        self.enc_out_channels = enc_out_channels
        resnet18 = models.video.r3d_18(pretrained=True)
        modules = list(resnet18.children())[:-1]
        resnet18 = nn.Sequential(*modules)
        self.model = resnet18

    def forward(self, x):
        with open("resnet_size_check.txt", "w") as fp:
            fp.write(str(x.shape))
        return self.model(x).squeeze()

