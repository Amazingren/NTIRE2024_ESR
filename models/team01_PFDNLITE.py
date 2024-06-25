# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalAttention(nn.Module):
    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            torch.nn.MaxPool2d(kernel_size = 7,stride = 3, padding=0),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        g = torch.sigmoid(x[:,:1])
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return w * g * x

class PFDB_Lite(nn.Module):
    def __init__(self,
                  channels):
        super(PFDB_Lite, self).__init__()
        self.attn = LocalAttention(channels)
        self.conv1 = nn.Conv2d(48,48,3,1,1)
        self.conv2 = nn.Conv2d(24,24,3,1,1)
        self.conv3 = nn.Conv2d(48,48,3,1,1)
        self.relu = nn.SiLU(True)

    def forward(self, x):
        x = self.attn(x)
        res = x
        x = self.relu(self.conv1(x))
        x[:,:24] = self.relu(self.conv2(x[:,:24]))
        x = self.relu(self.conv3(x))
        x = x + res
        return x

class PFDB_Lite_Prune(nn.Module):
    def __init__(self,
                  channels):
        super(PFDB_Lite_Prune, self).__init__()
        self.attn = LocalAttention(channels)
        self.conv1 = nn.Conv2d(48,24,3,1,1)
        self.conv3 = nn.Conv2d(48,48,3,1,1)
        self.relu = nn.SiLU(True)

    def forward(self, x):
        x = self.attn(x)
        res = x.clone()
        x[:, :24] = self.relu(self.conv1(res))
        x = self.relu(self.conv3(x))
        x = x + res
        return x

class PFDN_Lite(nn.Module):
    def __init__(self,
                  scale=4,
                  in_channels=3,
                  out_channels=3,
                  feature_channels=48):

        super(PFDN_Lite, self).__init__()
        self.scale = scale

        self.head = nn.Conv2d(in_channels, feature_channels, 3, 1, 1)

        self.block1 = PFDB_Lite(48)
        self.block2 = PFDB_Lite_Prune(48)
        self.block3 = PFDB_Lite_Prune(48)
        self.block4 = PFDB_Lite(48)

        self.tail = nn.Sequential(
            nn.Conv2d(feature_channels, out_channels * (scale ** 2), 3, 1, 1),
        )
        self.upsampler = nn.PixelShuffle(scale)


    def forward(self, x):
        shortcut = torch.repeat_interleave(x, 16, dim=1)
        x = self.head(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.tail(x) + shortcut
        x = self.upsampler(x)
        return x
