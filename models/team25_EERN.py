import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
def make_model(parent=False):
    return EERN()

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        c1 = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(c1)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        c4 = self.conv4(c3 + c1_)
        m = self.sigmoid(c4)
        return x * m

class ResidualBlock_ESA(nn.Module):
    '''
    ---Conv-ReLU-Conv-ESA +-
    '''
    def __init__(self, nf=32):
        super(ResidualBlock_ESA, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.ESA = ESA(nf, nn.Conv2d)
        # self.lrelu = nn.LeakyReLU(0.1)
        self.lrelu = nn.ReLU()

    def forward(self, x):
        out = (self.conv1(x))
        out = self.lrelu(out)
        out = (self.conv2(out))
        out = self.ESA(out)
        return out

class EERN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, upscale=4):
        super(EERN, self).__init__()
        nf = 84
        blocks = 4
        basic_block = functools.partial(ResidualBlock_ESA, nf=nf)
        self.recon_trunk = make_layer(basic_block, blocks)
        self.head = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.tail = nn.Conv2d(nf, out_nc * upscale * upscale, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        ##################
        # self.relu = nn.ReLU()
        self.finetune = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # 初始化finetune权重为0
        init.constant_(self.finetune.weight, 0)
        if self.finetune.bias is not None:
            init.constant_(self.finetune.bias, 0)
        ##################

    def forward(self, x):
        fea = self.head(x)
        out = fea
        layer_names = self.recon_trunk._modules.keys()
        for layer_name in layer_names:
            fea = self.recon_trunk._modules[layer_name](fea)
        
        #########################
        fea = self.finetune(fea)
        # noise = torch.randn_like(fea) * 8e-3
        # fea = fea + noise
        # fea = self.relu(fea)
        ######################
        out = fea + out
        out = self.pixel_shuffle(self.tail(out))
        return out
