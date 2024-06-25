import torch.nn as nn
import torch.nn.functional as F


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int,
                 upsample_kernel: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class ESA(nn.Module):
    def __init__(self, esa_channels=16, n_feats=30):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, esa_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(esa_channels, esa_channels, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(esa_channels, esa_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(esa_channels, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        c4 = self.conv4(c3 + c1_)
        m = self.sigmoid(c4)
        return x * m


class RLFB_Prune(nn.Module):
    def __init__(self,
                 channels,
                 prune_channels=None,
                 ):
        super().__init__()
        self.c1_r = nn.Conv2d(channels, prune_channels[0], 3, padding=1)
        self.c2_r = nn.Conv2d(prune_channels[0], prune_channels[1], 3, padding=1)
        self.c3_r = nn.Conv2d(prune_channels[1], channels, 3, padding=1)

        self.c5 = nn.Conv2d(channels, channels, 1)
        self.esa = ESA(16, channels)

        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)

    def forward(self, x):
        out = self.act(self.c1_r(x))
        out = self.act(self.c2_r(out))
        out = self.act(self.c3_r(out))

        out = out + x
        out = self.esa(self.c5(out))
        return out


class SlimRLFN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=30,
                 num_block=6,
                 prune_channels=None,
                 upscale=4):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, feature_channels, 3, padding=1)
        if prune_channels is None:
            prune_channels = [feature_channels for _ in range(2 * num_block)]
        else:
            assert len(prune_channels) == 2 * num_block, f'{prune_channels} is illegal.'

        self.block_list = nn.ModuleList([
            RLFB_Prune(feature_channels, prune_channels=prune_channels[i * 2:2 * (i + 1)])
            for i in range(num_block)])

        self.conv_2 = nn.Conv2d(feature_channels, feature_channels, 3, padding=1)
        self.upsampler = PixelShufflePack(feature_channels, out_channels,
                                          upscale, upsample_kernel=3)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out = out_feature
        for module in self.block_list:
            out = module(out)

        out_low_resolution = self.conv_2(out) + out_feature
        output = self.upsampler(out_low_resolution)

        return output
