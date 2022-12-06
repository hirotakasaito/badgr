import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convs = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2), nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2), nn.ReLU())
    def forward(self, x):
        return self.convs(x)

class Decoder(nn.Module):
    def __init__(self):
        self.deconvs = nn.Sequential(nn.ConvTranspose2d(256, 256, 2, stride=2), nn.BatchNorm2d(256), nn.ReLU(),
                                    nn.ConvTranspose2d(256, 128, 2, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
                                    nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
                                    nn.ConvTranspose2d(32, 16, 2, stride=2), nn.BatchNorm2d(16), nn.ReLU(),
                                    nn.ConvTranspose2d(16, 3, 2, stride=2), nn.Tanh())

    def foward(self, x):
        return self.deconvs(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = Encoder.convs()
        self.decoder = Decoder.deconvs()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x