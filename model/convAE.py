"""
@Author: MaraPapMann
@Description: The definition & structure of "GAN-based Auto-augmenter" & "Binary Classifier"
"""
import functools
from typing import Any
import MaranusHanusCitrus.torchlib as torchlib
import torch as T
from torch import nn
import torchvision as TV


def _get_norm_layer_2d(norm:str)->Any:
    """
    To get a norm layer by the input string.
    @Params:
        norm: Name of the norm layer;
    @Return:
        A type of norm layer.
    """
    if norm == 'none':
        return torchlib.Identity
    elif norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm == 'layer_norm':
        return lambda num_features: nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError


class ConvGenerator(nn.Module):

    def __init__(self,
                 output_channels=3,
                 dim=64,
                 n_samplings=5,
                 norm='batch_norm'):
        super().__init__()

        Norm = _get_norm_layer_2d(norm)

        def dconv_norm_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding,
                                   bias=False or Norm == torchlib.Identity),
                Norm(out_dim),
                nn.ReLU()
            )

        layers = []

        # 1: 1x1 -> 4x4
        d = min(dim * 2 ** (n_samplings - 1), dim * 16)
        # layers.append(dconv_norm_relu(input_dim, d, kernel_size=4, stride=1, padding=0))

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
        for i in range(n_samplings - 1):
            d_last = d
            d = min(dim * 2 ** (n_samplings - 2 - i), dim * 16)
            layers.append(dconv_norm_relu(d_last, d, kernel_size=4, stride=2, padding=1))

        # layers.append(nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
        layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
        layers.append(nn.Conv2d(d, output_channels, kernel_size=5, stride=1, padding=2))

        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)


    def forward(self, z):
        x = self.net(z)
        return x


class ConvDiscriminator(nn.Module):

    def __init__(self,
                 input_channels=3,
                 dim=64,
                 n_downsamplings=6,
                 norm='batch_norm'):
        super().__init__()

        Norm = _get_norm_layer_2d(norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding,
                          bias=False or Norm == torchlib.Identity),
                Norm(out_dim),
                nn.LeakyReLU(0.2)
            )

        layers = []  # Create a list for appending layers

        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = dim
        layers.append(nn.Conv2d(input_channels, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        for i in range(n_downsamplings - 1):
            d_last = d
            d = min((dim * 2 ** (i + 1), dim * 128))
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))

        # 2: logit
        layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        y = self.net(x)
        return y


class GANAutoEncoder(nn.Module):

    def __init__(self,
                n_channels:int=3,
                dim:int=64,
                n_samplings:int=5,
                norm:str='batch_norm'):
        super().__init__()

        Norm = _get_norm_layer_2d(norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding,
                          bias=False or Norm == torchlib.Identity),
                Norm(out_dim),
                nn.LeakyReLU(0.2)
            )
        
        def dconv_norm_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding,
                                   bias=False or Norm == torchlib.Identity),
                Norm(out_dim),
                nn.ReLU()
            )
        
        layers = []

        # ============================================
        # =                  Decoder                 =
        # ============================================

        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        layers.append(nn.Conv2d(n_channels, dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        d = dim
        for i in range(n_samplings - 1):
            d_last = d
            d = min(dim * 2 ** (i + 1), dim * 128)
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))
        
        # ============================================
        # =                  Encoder                 =
        # ============================================

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
        d = min(dim * 2 ** (n_samplings - 1), dim * 128)
        for i in range(n_samplings - 1):
            d_last = d
            d = min(dim * 2 ** (n_samplings - 2 - i), dim * 128)
            layers.append(dconv_norm_relu(d_last, d, kernel_size=4, stride=2, padding=1))

        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
        layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
        layers.append(nn.Conv2d(d, n_channels, kernel_size=5, stride=1, padding=2))
        layers.append(nn.Tanh())

        # Complete the auto-encoder
        self.net = nn.Sequential(*layers)
    

    def forward(self, x):
        y = self.net(x)
        return y
