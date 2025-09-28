import torch
import torch.nn as nn

from .generator import Generator
from .discriminator import Discriminator

class CycleGAN(nn.Module):
    def __init__(self, img_channels=1, num_features=64, num_residuals=9):
        super(CycleGAN, self).__init__()
        self.gen_TB2MBOD = Generator(img_channels, num_features, num_residuals)
        self.gen_MBOD2TB = Generator(img_channels, num_features, num_residuals)
        self.discriminator_TB = Discriminator(img_channels)
        self.discriminator_MBOD = Discriminator(img_channels)

    def forward_cons(self, x):
        fake_MBOD = self.gen_TB2MBOD(x)
        fake_TB = self.gen_MBOD2TB(fake_MBOD)
        return fake_TB, fake_MBOD

    def backward_cons(self, x):
        fake_TB = self.gen_TB2MBOD(x)
        fake_MBOD = self.gen_MBOD2TB(fake_TB)
        return fake_TB, fake_MBOD   

    def discriminate(self, x):
        real_MBOD_preds = self.discriminator_MBOD(x[0])
        real_TB_preds = self.discriminator_TB(x[1])
        return real_TB_preds, real_MBOD_preds