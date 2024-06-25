from audioop import bias

from einops import rearrange

"""
不进行修改
"""
# import thop
# from ptflops import get_model_complexity_info

"""
加上大号dense连接
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.init import trunc_normal_


# from basicsr.utils.registry import ARCH_REGISTRY


def get_local_weights(residual, ksize, padding):
    pad = padding
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
    unfolded_residual = residual_pad.unfold(2, ksize, 3).unfold(3, ksize, 3)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight


class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, input=(256, 256)):
        super(ESA, self).__init__()
        f = num_feat // 4
        self.num_feat = num_feat
        self.input = input
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.conv2_0 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_1 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_2 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_3 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.maxPooling_0 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.maxPooling_1 = nn.MaxPool2d(kernel_size=5, stride=3)
        self.maxPooling_2 = nn.MaxPool2d(kernel_size=7, stride=3, padding=1)
        self.conv_max_0 = conv(f, f, kernel_size=3)
        self.conv_max_1 = conv(f, f, kernel_size=3)
        self.conv_max_2 = conv(f, f, kernel_size=3)
        self.var_3 = get_local_weights

        self.conv3_0 = conv(f, f, kernel_size=3)
        self.conv3_1 = conv(f, f, kernel_size=3)
        self.conv3_2 = conv(f, f, kernel_size=3)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()
        # self.norm = nn.BatchNorm2d(num_feat)
        # self.seita = nn.Parameter(torch.normal(mean=0.5, std=0.01, size=(1, 1, 1)))
        # self.keci = nn.Parameter(torch.normal(mean=0.5, std=0.01, size=(1, 1, 1)))
        #
        # self.alpha = nn.Parameter(torch.normal(mean=0.25, std=0.01, size=(1,1,1)))
        # self.beta = nn.Parameter(torch.normal(mean=0.25, std=0.01, size=(1,1,1)))
        # self.gama = nn.Parameter(torch.normal(mean=0.25, std=0.01, size=(1,1,1)))
        # self.omega = nn.Parameter(torch.normal(mean=0.25, std=0.01, size=(1,1,1)))

    def forward(self, input):
        c1_ = self.conv1(input)  # channel squeeze
        temp = self.conv2_0(c1_)
        c1_0 = self.maxPooling_0(temp)  # strided conv 3
        c1_1 = self.maxPooling_1(self.conv2_1(c1_))  # strided conv 5
        c1_2 = self.maxPooling_2(self.conv2_2(c1_))  # strided conv 7
        c1_3 = self.var_3(self.conv2_3(c1_), 7, padding=1)  # strided local-var 7

        v_range_0 = self.conv3_0(self.GELU(self.conv_max_0(c1_0)))
        v_range_1 = self.conv3_1(self.GELU(self.conv_max_1(c1_1)))
        v_range_2 = self.conv3_2(self.GELU(self.conv_max_2(c1_2 + c1_3)))

        c3_0 = F.interpolate(v_range_0, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        c3_1 = F.interpolate(v_range_1, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        c3_2 = F.interpolate(v_range_2, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)

        cf = self.conv_f(c1_)
        c4 = self.conv4((c3_0 + c3_1 + c3_2 + cf))
        m = self.sigmoid(c4)

        return input * m


class MDSA(nn.Module):
    def __init__(self, c_dim, conv, input):
        super().__init__()
        self.body = nn.Sequential(
            nn.GELU(),
            ESA(c_dim, conv, input),
            nn.Conv2d(c_dim, c_dim, (3, 3), padding='same', groups=c_dim))
        self.input = input
        self.c_dim = c_dim

    def forward(self, x):
        sa_x = self.body(x)
        sa_x += x
        return sa_x


def window_partition(x, window_size):
    """
     将feature map按照window_size划分成一个个没有重叠的window
     Args:
         x: (B, H, W, C)
         window_size (int): window size(M)

     Returns:
         windows: (num_windows*B, window_size, window_size, C)
     """
    B, H, W, C = x.shape
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    B, H_p, W_p, C = x.shape
    x = x.view(B, H_p // window_size, window_size, W_p // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    pad_true = bool(pad_b + pad_r + pad_l + pad_t)
    return x, pad_true, H_p, W_p


def window_reverse(windows, window_size, H, W):
    """
        将一个个window还原成一个feature map
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size(M)
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
    """
    # print("H:", H)
    # print("W:", W)
    # print("window shape", windows.shape)

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CCALayer(nn.Module):
    def __init__(self, channel, reduction, input):
        super(CCALayer, self).__init__()
        self.c_dim = channel
        self.input = input
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ECCA(nn.Module):
    def __init__(self, c_dim, reduction, input):
        super().__init__()
        self.input = input
        self.c_dim = c_dim
        self.body = nn.Sequential(nn.Conv2d(c_dim, c_dim, (1, 1), padding='same'),
                                  nn.GELU(),
                                  CCALayer(c_dim, reduction, input),
                                  nn.Conv2d(c_dim, c_dim, (3, 3), padding='same', groups=c_dim))

    def forward(self, x):
        ca_x = self.body(x)
        ca_x += x
        return ca_x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 input=(64, 64)):
        super().__init__()
        self.input = input

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwc1 = Dconv_for_MLP(hidden_features, hidden_features, 3, 'same', input)
        # self.dwc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,groups=hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features

    def forward(self, x):
        x = self.dwc1(self.fc1(x))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.act(x)
        x = self.drop(x)
        return x


class Dconv_basic(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding, input):
        super().__init__()
        self.dim = in_dim // 4
        self.input = input
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.conv2_1 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_2 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size * 2 - 1, kernel_size), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_3 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size * 2 - 1), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_4 = [
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding, groups=in_dim // 4),
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding, groups=in_dim // 4)]
        self.conv2_4 = nn.Sequential(*self.conv2_4)
        # self.conv1x1_2 = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.act = nn.GELU()

    def forward(self, input, flag=False):
        out = self.conv1(input)
        out = torch.chunk(out, 4, dim=1)
        s1 = self.act(self.conv2_1(out[0]))
        s2 = self.act(self.conv2_2(out[1] + s1))
        s3 = self.act(self.conv2_3(out[2] + s2))
        s4 = self.act(self.conv2_4(out[3] + s3))
        out = torch.cat([s1, s2, s3, s4], dim=1) + input
        out = self.act(out)
        return out

class Dconv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding, input):
        super().__init__()
        self.dim = in_dim // 4
        self.input = input
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.conv2_1 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_2 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size * 2 - 1, kernel_size), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_3 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size * 2 - 1), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_4 = [
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding, groups=in_dim // 4),
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding, groups=in_dim // 4)]
        self.conv2_4 = nn.Sequential(*self.conv2_4)
        self.conv1x1_2 = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.act = nn.GELU()

    def forward(self, input, flag=False):
        out = self.conv1(input)
        out = torch.chunk(out, 4, dim=1)
        s1 = self.conv2_1(out[0])
        s2 = self.conv2_2(out[1] + s1)
        s3 = self.conv2_3(out[2] + s2)
        s4 = self.conv2_4(out[3] + s3)
        out = torch.cat([s1, s2, s3, s4], dim=1)
        out = self.conv1x1_2(self.act(out))
        return out


class Dconv_for_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding, input):
        super().__init__()
        self.input = input
        self.dim = in_dim // 4
        self.conv2_1 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_2 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size * 2 - 1, kernel_size), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_3 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size * 2 - 1), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_4 = [
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding, groups=in_dim // 4),
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding, groups=in_dim // 4)]
        self.conv2_4 = nn.Sequential(*self.conv2_4)
        self.act = nn.GELU()

    def forward(self, input):
        out = torch.chunk(input, 4, dim=1)

        s1 = self.act(self.conv2_1(out[0]))
        s2 = self.act(self.conv2_2(out[1] + s1))
        s3 = self.act(self.conv2_3(out[2] + s2))
        s4 = self.act(self.conv2_4(out[3] + s3))
        out = torch.cat([s1, s2, s3, s4], dim=1) + input
        return out


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return img


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, shift_size, dim_out=None, num_heads=1, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.idx = idx
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.shift_size = shift_size
        # self.split_size = resolution // 4
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == 0:
            H_sp, W_sp = 32, 8
        elif idx == 1:
            W_sp, H_sp = 32, 8
        elif idx == 2:
            W_sp, H_sp = 32, 8
        else:
            W_sp, H_sp = 32, 32
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj_qk = nn.Conv2d(dim, 2 * dim, 1)
        self.proj_q_5x5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj_k_5x5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj_v = nn.Conv2d(dim, dim, 1)
        self.lepe_v = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)

    def im2cswin(self, x):
        B, C, H, W = x.shape
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x):
        """
        x: B L C
        """
        B, C, H, W = x.shape

        pad_l = pad_t = 0
        pad_r = (self.W_sp - W % self.W_sp) % self.W_sp
        pad_b = (self.H_sp - H % self.H_sp) % self.H_sp
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, Hp, Wp = x.shape
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.H_sp//2, -self.W_sp//2), dims=(2, 3))

        else:
            x = x

        q, k = torch.chunk(self.proj_qk(x), 2, dim=1)
        v = self.proj_v(x)
        lepe = self.lepe_v(v)

        ### Img2Window

        q = self.im2cswin(self.proj_q_5x5(q) + lepe)
        k = self.im2cswin(self.proj_k_5x5(k) + lepe)

        # lepe = self.get_lepe(v)
        v = self.im2cswin(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C).contiguous()  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, Hp, Wp) + lepe  # B C H' W'
        # shift 还原
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.H_sp//2, self.W_sp//2), dims=(2, 3))
        else:
            x = x
        if pad_r or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()

        return x

    def flops(self, N, input_size):
        # calculate flops for 1 window with token length of N
        H, W = input_size

        flops = 0
        # qkv = self.qkv(x)
        flops += H * W * self.dim * 3 * self.dim
        # DWconv qkv
        flops += H * W * self.dim * 5 * 5 * 3
        # attn = (q @ k.transpose(-2, -1))
        flops += N * self.dim * H * W
        #  x = (attn @ v)
        flops += N * self.dim * H * W
        return flops


class Gated_Module(nn.Module):
    def __init__(self, dim, input):
        super().__init__()
        self.input = input
        self.dim = dim
        self.gated_reset = nn.Conv2d(dim * 2, dim, 1)
        self.gated_update = nn.Conv2d(dim * 2, dim, 1)
        self.gated_fusion = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x, h):
        r = torch.sigmoid(self.gated_reset(torch.cat([x, h], 1)))
        z = torch.sigmoid(self.gated_update(torch.cat([x, h], 1)))
        h_hat = torch.tanh(self.gated_fusion(torch.cat([x, h * r], 1)))
        out = (1. - z) * h + z * h_hat
        return out


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, input=(256, 256)):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input = input
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ration = mlp_ratio
        "shift_size must in 0-window_size"
        assert 0 <= self.shift_size < self.window_size
        "层归一化"
        # self.norm1 = norm_layer(dim)
        # self.attn1 = WinGlobalAttention(dim // 2, 1, input, 32) # 16*16
        self.attn1 = LePEAttention(dim // 2, window_size, 3, self.shift_size)  # 16*4
        # self.attn2 = LePEAttention(dim//2, window_size, 0)  #16*4
        self.attn3 = LePEAttention(dim // 2, window_size, 0, self.shift_size)
        # self.attn3 = LePEAttention(dim//2, window_size, 1)  #4*16
        self.attn4 = LePEAttention(dim // 2, window_size, 1, self.shift_size)   # 4*16
        # self.attn4 = WinGlobalAttention(dim // 2, 1, input, 32)  # 8*8
        self.attn2 = LePEAttention(dim // 2, window_size, 2, self.shift_size)  # 4*16
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop, input=input)
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.proj = nn.Conv2d(dim * 2, dim, 1)
        self.proj1 = nn.Conv2d(dim, dim * 2, 1)
        # self.proj_conv = nn.Conv2d(dim, dim, 3, 1, padding=1, groups=dim)
        self.gate_layer = Gated_Module(dim // 2, input)

    def forward(self, x):  # x: B,C,H,W

        B, C, H, W = x.shape  # x: B,C,H,W
        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        shortcut = x
        shift_x_norm = self.norm1(x)
        x_window = self.proj1(shift_x_norm)
        x_slices = torch.chunk(x_window, 4, dim=1)
        x1 = self.attn1(x_slices[0])
        x2 = self.attn2(self.gate_layer(x_slices[1], x1))
        x3 = self.attn3(self.gate_layer(x_slices[2], x2))
        x4 = self.attn4(self.gate_layer(x_slices[3], x3))

        attened_x = torch.cat([x1, x2, x3, x4], dim=1)
        del x1, x2, x3, x4
        # 从窗口还原到原来的大小

        x = self.proj(attened_x)+shortcut
        # x_reverse = x_reverse_spa+x_reverse_ca

        x = x + self.mlp(self.norm2(x))  # x: B,H,W,C
        return x  # x: B,C,H,W


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class Spatial_Attn(nn.Module):
    def __init__(self, c_dim, depth, windows_size, num_heads, input):
        super().__init__()
        self.input = input
        swin_body = []
        self.window_size = windows_size

        if depth % 2:
            shift_size = windows_size // 2
        else:
            shift_size = 0
        self.shift_size = shift_size
        swin_body.append(SwinTransformerBlock(c_dim, num_heads, window_size=windows_size, shift_size=shift_size,
                                              mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                                              act_layer=nn.GELU))
        self.swin_body = nn.Sequential(*swin_body)

    def forward(self, x):
        src = x
        _, _, H, W, = x.shape
        info_mix = self.swin_body(src)

        return info_mix


class Res_Spatial_Attn(nn.Module):
    def __init__(self, c_dim, depth, windows_size, num_heads, input):
        super().__init__()
        self.input = input
        self.c_dim = c_dim
        self.body = nn.Sequential(nn.GELU(),
                                  Spatial_Attn(c_dim, depth, windows_size, num_heads, input),
                                  nn.Conv2d(c_dim, c_dim, (3, 3), padding=1, groups=c_dim))

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def Pixelshuffle_Block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), (kernel_size, kernel_size), (stride, stride),
                     padding='same')
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)


class BSConvU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None, input=(256, 256)):
        super().__init__()
        self.input = input
        self.in_dim = in_channels
        self.out_dim = out_channels
        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode='reflect',
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class BasicLayer(nn.Module):
    def __init__(self, in_channels, reduction, RC_depth, RS_depth, depth, windows_size, num_heads, input):
        super(BasicLayer, self).__init__()
        kwargs = {'padding': 1}
        conv = ECCA
        self.depth = depth
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.input = input
        self.c_dim = in_channels
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = Dconv_basic(in_channels, in_channels, 3, "same", input)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = Dconv_basic(in_channels, in_channels, 3, "same", input)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = Dconv_basic(in_channels, in_channels, 3, "same", input)

        self.c4 = BSConvU(self.remaining_channels, self.dc, kernel_size=3, **kwargs, input=input)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        if depth == 0 or depth == 1 or depth == 2:
            self.esa = Res_Spatial_Attn(in_channels, depth, windows_size, num_heads, input)
        else:
            self.esa = MDSA(in_channels, BSConvU, input)
        self.cca = ECCA(in_channels, reduction=16, input=input)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2,distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        out_fused = self.esa(out)  # MDSA
        out_fused = self.cca(out_fused)  # ECCA
        return out_fused + input


# @ARCH_REGISTRY.register()
class HARN(nn.Module):
    def __init__(self, rgb_mean=[0.4488, 0.4371, 0.4040], upscale_factor=4, c_dim=20, reduction=16, Bsc_depth=4,
                 RS_depth=1, RC_depth=0, depth=1,
                 windows_size=32, num_heads=4, task=None, input=(64, 64)):
        super(HARN, self).__init__()
        self.body = []
        self.input = input
        self.c_dim = c_dim
        self.upscale_factor = upscale_factor
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.conv_shallow = nn.Conv2d(3, c_dim, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.body.extend(
            [BasicLayer(c_dim, reduction, RC_depth, RS_depth, depth, windows_size, num_heads, self.input) for depth in
             range(Bsc_depth)])
        self.conv_before_upsample = nn.Sequential(Dconv(c_dim, c_dim, 3, padding='same', input=input))
        self.upsample = nn.Sequential(Pixelshuffle_Block(c_dim, 3, upscale_factor=upscale_factor, kernel_size=3))
        self.bsc_layer = nn.Sequential(*self.body)
        self.c = nn.Conv2d(Bsc_depth * c_dim, c_dim, 1)
        # self.out = nn.Conv2d(3, 3, 3,1,1)
        # self.conv1x1_1 = nn.Conv2d(c_dim, c_dim, 1)
        # self.conv1x1_2 = nn.Conv2d(c_dim, c_dim, 1)
        # self.conv1x1_3 = nn.Conv2d(c_dim, c_dim, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = x - self.mean
        out_fea = self.conv_shallow(x)
        x1 = self.bsc_layer[0](out_fea)
        x2 = self.bsc_layer[1](x1)
        x3 = self.bsc_layer[2](x2)
        x4 = self.bsc_layer[3](x3)
        # x5 = self.bsc_layer[4](x4)
        # x6 = self.bsc_layer[4](x5)
        # x7 = self.bsc_layer[4](x6)
        # x8 = self.bsc_layer[4](x7)
        out_B = self.c(torch.cat([x1, x2, x3, x4], dim=1))
        out_lr = self.conv_before_upsample(out_B) + out_fea

        output = self.upsample(out_lr) + self.mean

        return output


if __name__ == '__main__':
    window_size = 8
    upscale = 4
    height = 256
    width = 256
    model = HARN(rgb_mean=(0.4040, 0.4371, 0.4488), upscale_factor=upscale, input=(height, width))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    x = torch.randn((1, 3, 256, 256))
    # a = time.time()
    # x = model(x)
    # b = time.time()
    # print(x.shape)
    # print(b-a)
    model.cuda()
    out = model(x.cuda())
    print(out.shape)

    iterations = 100  # 重复计算的轮次

    device = torch.device("cuda:0")
    model.to(device)

    random_input = x.to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(50):
        with torch.no_grad():
            _ = model(random_input)

    # 测速
    times = torch.zeros(iterations)  # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(random_input)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[iter] = curr_time
            # print(curr_time)

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

    print('最大显存', torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2)
    # print(torch.cuda.memory_summary())

    from fvcore.nn import FlopCountAnalysis

    input_fake = torch.rand(1, 3, 256, 256).to(device)
    flops = FlopCountAnalysis(model, input_fake).total()
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))