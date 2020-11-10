import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .base_color import *


class BaseNet(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(BaseNet, self).__init__()
        # O = W-(dia*(K-1)-1+2P / S  + 1
        # (128-5 +2X1 +2) / 2  = 129/2 - 64
        k_size = 5
        stride = 2
        padding = 2

        ## Encoder
        self.encoder = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(1, 16, kernel_size=k_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(True),
            norm_layer(16),

            # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=k_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(True),
            norm_layer(32),

            # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=k_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(True),
            norm_layer(64),

            # 16 -> 8
            nn.Conv2d(64, 128, kernel_size=k_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(True),
            norm_layer(128),

            # 8 -> 4
            nn.Conv2d(128, 256, kernel_size=k_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(True),
            norm_layer(256)
        )

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Linear(256 * 4 * 4, 2)

    def forward(self, input_l):
        conv = self.encoder(self.normalize_l(input_l))
        flatten = torch.flatten(conv, start_dim=1)
        out_reg = self.model_out(self.softmax(flatten))

        return out_reg
