import random

from loader import SnakeGameDataset as dataset

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms

import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        # Encoder
        self.enc_layer_1 = nn.Conv2d(1, 8, kernel_size=9, stride=2)
        self.enc_layer_2 = nn.Conv2d(8, 16, kernel_size=7)
        self.enc_layer_3 = nn.Conv2d(16, 32, kernel_size=5)
        #self.enc_linear_1 = nn.Linear(4*4*20, 50)
        self.enc_linear_1 = nn.Linear(12*12*32, 2300)
        self.enc_linear_2 = nn.Linear(2300, 1000)
        self.enc_linear_3 = nn.Linear(1000, self.code_size)

        # Decoder
        self.dec_linear_1 = nn.Linear(self.code_size, 1000)
        self.dec_linear_2 = nn.Linear(1000, 2300)
        self.dec_linear_3 = nn.Linear(2300, 12*12*32)
        self.dec_layer_1 = nn.ConvTranspose2d(32, 16, kernel_size=5)
        self.dec_layer_2 = nn.ConvTranspose2d(16, 8, kernel_size=7)
        self.dec_layer_3 = nn.ConvTranspose2d(8, 1, kernel_size=9, stride=2, output_padding=1)

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code

    def encode(self, images):
        code = F.selu(self.enc_layer_1(images))

        code = F.selu(self.enc_layer_2(code))

        code = F.selu(self.enc_layer_3(code))

        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = F.selu(self.enc_linear_2(code))
        code = self.enc_linear_3(code)

        return code

    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.selu(self.dec_linear_2(out))
        out = F.selu(self.dec_linear_3(out))
        out = out.view([code.size(0), 32, 12, 12])
        out = F.selu(self.dec_layer_1(out))
        out = F.selu(self.dec_layer_2(out))
        out = F.selu(self.dec_layer_3(out))
        # out = out.view([code.size(0), 1, IMAGE_WIDTH, IMAGE_HEIGHT])

        return out
