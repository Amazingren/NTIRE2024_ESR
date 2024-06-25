import torch.nn as nn
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True,
               rep='plain'):
    if rep == 'plain' :
        kernel_size = _make_pair(kernel_size)
        padding = (int((kernel_size[0] - 1) / 2),
                int((kernel_size[1] - 1) / 2))
        return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)
    else:
        raise NotImplementedError(
                f'rep type not supported')


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):

    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


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
                       upscale_factor=2,
                       kernel_size=3,
                       bias=True, rep='plain'):
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size, bias=bias, rep=rep)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):

    def __init__(self, esa_channels, n_feats, conv, bias=True, rep='plain'):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1, bias=bias, rep=rep)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0, bias=bias)
        self.conv3 = conv(f, f, kernel_size=3, bias=bias, rep=rep)
        self.conv4 = conv(f, n_feats, kernel_size=1, bias=bias, rep=rep)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        c1 = F.max_pool2d(c1, kernel_size=7, stride=3)
        c1 = self.conv3(c1)
        c1 = F.interpolate(c1, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        c1 = self.conv4(c1 + c1_)
        m = self.sigmoid(c1)
        return x * m



class RRFB(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16,
                 bias=True,
                 rep='plain'):
        super(RRFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3, bias=bias, rep=rep)
        self.c3_r = conv_layer(mid_channels, in_channels, 3, bias=bias, rep=rep)

        self.esa = ESA(esa_channels, out_channels, conv_layer, bias=bias, rep=rep)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c3_r(out))

        out = self.esa(out)

        return out



class R2Net(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=50,
                 upscale=4,
                 bias=True,
                 rep='plain'):
        super(R2Net, self).__init__()

        self.conv_1 = conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3, bias=bias, rep=rep)

        self.block_1 = RRFB(feature_channels, bias=bias, rep=rep)
        self.block_2 = RRFB(feature_channels, bias=bias, rep=rep)
        self.block_3 = RRFB(feature_channels, bias=bias, rep=rep)
        self.block_4 = RRFB(feature_channels, bias=bias, rep=rep)

        self.conv_2 = conv_layer(feature_channels,
                                   feature_channels,
                                   kernel_size=1, bias=bias, rep=rep)

        self.upsampler = pixelshuffle_block(feature_channels,
                                          out_channels,
                                          upscale_factor=upscale, bias=bias, rep=rep)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out = self.block_1(out_feature)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)


        out_low_resolution = self.conv_2(out) + out_feature
        output = self.upsampler(out_low_resolution)

        return output
