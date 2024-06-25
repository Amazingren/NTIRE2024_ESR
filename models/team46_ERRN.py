from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2):
    """
    Upsample features according to `upscale_factor`.
    """
    conv =nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1, bias=True)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class SimpleSpatialAttention(nn.Module):
    def __init__(self, in_planes):
        super(SimpleSpatialAttention, self).__init__()

        self.sa_conv = nn.Conv2d(in_planes, 1, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        return x*self.sigmoid(self.sa_conv(x))


class RB1(nn.Module):
    def __init__(self, n_feats):
        super(RB1, self).__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)
        return out


class RB2(nn.Module):
    def __init__(self, n_feats):
        super(RB2, self).__init__()
        self.rep_conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv2(x)
        return out


class RepB(nn.Module):

    def __init__(self,
                 in_channels):
        super(RepB, self).__init__()
        self.rep1_1 = RB1(in_channels)
        self.rep1_2 = RB1(in_channels)
        self.rep2_1 = RB2(in_channels)
        self.rep2_2 = RB2(in_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.ssa = SimpleSpatialAttention(in_channels)

    def forward(self, x):
        out = self.rep1_1(x)
        out = self.act(out)
        out = self.rep1_2(out)
        out = self.act(out)
        out = self.rep2_1(out)
        out = self.act(out)
        out = self.rep2_2(out)
        return self.ssa(out+x)

class ERRN(nn.Module):
    """
    Enhanced Rep Residual  Network (ERRN)
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=48,
                 upscale=4):
        super(ERRN, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels,feature_channels,kernel_size=3,stride=1,padding=1,bias=True)
        self.block_1 = RepB(feature_channels)
        self.block_2 = RepB(feature_channels)
        self.block_3 = RepB(feature_channels)
        self.block_4 = RepB(feature_channels)
        self.block_5 = RepB(feature_channels)
        self.block_6 = RepB(feature_channels)
        self.conv_2 = nn.Conv2d(feature_channels,feature_channels,kernel_size=3,stride=1,padding=1,bias=True)
        self.upsampler = pixelshuffle_block(feature_channels,out_channels,upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)

        out_low_resolution = self.conv_2(out_b6) + out_feature
        output = self.upsampler(out_low_resolution)
        return output