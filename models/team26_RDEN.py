
import torch.nn as nn
import torch
import torch.nn.functional as F


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.
    
    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
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
    
def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), 
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)

def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3,
                       pruning_factor = 0,
                       bias = True):

    in_channels = int(in_channels * (1-pruning_factor))
    out_channels = out_channels
    
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size,
                      bias = bias)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):
    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)  
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m
        

class ECB(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type):
        super(ECB, self).__init__()
        self.act = nn.GELU()
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.conv_infer = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3,stride=1, padding=1, bias = False)

    def forward(self, x):
        y = self.conv_infer(x)
        y = self.act(y)
        return y

class SRRB(nn.Module):

    def __init__(self,
                 in_channels, 
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16,
                 act_type='grelu',
                 ):
        super(SRRB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = ECB(in_channels, mid_channels, act_type)
        self.c2_r = ECB(mid_channels, mid_channels, act_type)
        self.c3_r = ECB(mid_channels, in_channels, act_type)
       
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

    def forward(self, x):
        out = (self.c1_r(x))
        out = (self.c2_r(out))
        out = (self.c3_r(out))
        out = out + x         
        out = self.esa(out)
        return out



class RDEN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=37,
                 mid_channels=37,
                 upscale_factor=4):
        super(RDEN, self).__init__()

        self.conv_1 = conv_layer(in_channels,
                                           feature_channels,
                                           kernel_size=3,
                                           bias=False)  

        self.block_1 = SRRB(feature_channels, mid_channels, act_type='grelu') 
        self.block_2 = SRRB(feature_channels, mid_channels, act_type='grelu')
        self.block_3 = SRRB(feature_channels, mid_channels, act_type='grelu')
        self.block_4 = SRRB(feature_channels, mid_channels, act_type='grelu')

        self.conv_2 = conv_layer(feature_channels,
                                           feature_channels,
                                           kernel_size=3,
                                           bias=False)  

        self.upsampler_x4 = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale_factor,bias=False) 


       
    def forward(self, x):
        out_feature = self.conv_1(x)
        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_low_resolution = self.conv_2(out_b4) + out_feature
        output_x4 = self.upsampler_x4(out_low_resolution)  
        return output_x4



    
    

