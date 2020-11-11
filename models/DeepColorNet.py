import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .base_color import *


class DeepColorNet(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(DeepColorNet, self).__init__()
        # O = W-(dia*(K-1)-1+2P / S  + 1
        # (128-5 +2X1 +2) / 2  = 129/2 - 64
        k_size = 5
        stride = 2
        padding = 2
        out_pad = 1
        ## Encoder
        self.encoder = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(1, 16, kernel_size=k_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(True),
            norm_layer(16), # batch normalization

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

        ## Decoder
        # O = (W-1)*S - 2P +dia*(K-1)+ outP +1
        self.decoder = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose2d(256, 256, kernel_size=k_size, stride=stride, padding=padding, output_padding=out_pad,
                               bias=True),
            nn.ReLU(True),
            norm_layer(256),

            # 8 -> 16
            nn.ConvTranspose2d(256, 128, kernel_size=k_size, stride=stride, padding=padding, output_padding=out_pad,
                               bias=True),
            nn.ReLU(True),
            norm_layer(128),

            # 16 -> 32
            nn.ConvTranspose2d(128, 128, kernel_size=k_size, stride=stride, padding=padding, output_padding=out_pad,
                               bias=True),
            nn.ReLU(True),
            norm_layer(128),

            # 32 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=k_size, stride=stride, padding=padding, output_padding=out_pad,
                               bias=True),
            nn.ReLU(True),
            norm_layer(64),

            # 64 -> 128
            nn.ConvTranspose2d(64, 32, kernel_size=k_size, stride=stride, padding=padding, output_padding=out_pad,
                               bias=True),
            nn.ReLU(True),
            norm_layer(32),

            # map to ab image
            nn.Softmax(dim=1),
            # keep same image size
            nn.Conv2d(32, 2, kernel_size=k_size, stride=1, padding=padding)
        )

    def forward(self, input_l):
        conv = self.encoder(self.normalize_l(input_l))
        upsample = self.decoder(conv)

        return upsample
