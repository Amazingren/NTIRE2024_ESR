## NTIRE 2024 Efficient Super-resolution ECNU_MViC 
## IFADNet: Intermittent Feature Aggregation with Distillation for Efficient Super-Resolution
## Author: Bohan Jia

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class PixelShuffleDirect(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch):
        super(PixelShuffleDirect, self).__init__()
        self.upsampleOneStep = UpsampleOneStep(scale, num_feat, num_out_ch, input_resolution=None)

    def forward(self, x):
        return self.upsampleOneStep(x)


class BSConvU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        with_ln=False,
        bn_kwargs=None,
    ):
        super().__init__()

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
            padding_mode="reflect",
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
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
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        if self.type == "conv1x1-sobelx":
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == "conv1x1-sobely":
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == "conv1x1-laplacian":
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError("the type of seqconv is not supported!")

    def forward(self, x):
        y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
        # explicitly padding with bias
        y0 = F.pad(y0, (1, 1, 1, 1), "constant", 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        # conv-3x3
        y1 = F.conv2d(
            input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes
        )
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None
        tmp = self.scale * self.mask
        k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
        for i in range(self.out_planes):
            k1[i, i, :, :] = tmp[i, 0, :, :]
        b1 = self.bias
        # re-param conv kernel
        RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
        # re-param conv bias
        RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
        RB = (
            F.conv2d(input=RB, weight=k1).view(
                -1,
            )
            + b1
        )
        return RK, RB


class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, use_atten=False, deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.use_atten = use_atten
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.deploy or self.use_atten == False:
            self.rbr_reparam = conv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
            )
        else:
            self.rbr_3x3_branch = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                padding_mode="zeros",
            )
            self.rbr_1x1_3x3_branch_1x1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * in_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=(0, 0),
                dilation=1,
                groups=1,
                padding_mode="zeros",
                bias=False,
            )
            self.rbr_1x1_3x3_branch_3x3 = nn.Conv2d(
                in_channels=2 * in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                padding_mode="zeros",
                bias=False,
            )
            self.rbr_conv1x1_sbx_branch = SeqConv3x3("conv1x1-sobelx", self.in_channels, self.out_channels)
            self.rbr_conv1x1_sby_branch = SeqConv3x3("conv1x1-sobely", self.in_channels, self.out_channels)
            self.rbr_conv1x1_lpl_branch = SeqConv3x3("conv1x1-laplacian", self.in_channels, self.out_channels)

    def forward(self, inputs):
        if self.deploy or self.use_atten == False:
            return self.rbr_reparam(inputs)
        else:
            return (
                self.rbr_3x3_branch(inputs)
                + self.rbr_1x1_3x3_branch_3x3(self.rbr_1x1_3x3_branch_1x1(inputs))
                + inputs
                + self.rbr_conv1x1_sbx_branch(inputs)
                + self.rbr_conv1x1_sby_branch(inputs)
                + self.rbr_conv1x1_lpl_branch(inputs)
            )

    def switch_to_deploy(self):
        if self.use_atten:
            kernel, bias = self.get_equivalent_kernel_bias()
            self.rbr_reparam = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            )
            self.rbr_reparam.weight.data = kernel
            self.rbr_reparam.bias.data = bias
            self.__delattr__("rbr_3x3_branch")
            self.__delattr__("rbr_1x1_3x3_branch_1x1")
            self.__delattr__("rbr_1x1_3x3_branch_3x3")
            self.__delattr__("rbr_conv1x1_sbx_branch")
            self.__delattr__("rbr_conv1x1_sby_branch")
            self.__delattr__("rbr_conv1x1_lpl_branch")
            self.deploy = True

    def get_equivalent_kernel_bias(self):
        # 3x3 branch
        kernel_3x3, bias_3x3 = self.rbr_3x3_branch.weight.data, self.rbr_3x3_branch.bias.data

        # 1x1+3x3 branch
        kernel_1x1_3x3_fuse = self._fuse_1x1_3x3_branch(
            self.rbr_1x1_3x3_branch_1x1, self.rbr_1x1_3x3_branch_3x3
        )
        # identity branch
        device = kernel_1x1_3x3_fuse.device  # just for getting the device
        kernel_identity = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device)
        for i in range(self.out_channels):
            kernel_identity[i, i, 1, 1] = 1.0

        kernel_1x1_sbx, bias_1x1_sbx = self.rbr_conv1x1_sbx_branch.rep_params()
        kernel_1x1_sby, bias_1x1_sby = self.rbr_conv1x1_sby_branch.rep_params()
        kernel_1x1_lpl, bias_1x1_lpl = self.rbr_conv1x1_lpl_branch.rep_params()

        return (
            kernel_3x3
            + kernel_1x1_3x3_fuse
            + kernel_identity
            + kernel_1x1_sbx
            + kernel_1x1_sby
            + kernel_1x1_lpl,
            bias_3x3 + bias_1x1_sbx + bias_1x1_sby + bias_1x1_lpl,
        )


    def _fuse_1x1_3x3_branch(self, conv1, conv2):
        weight = F.conv2d(conv2.weight.data, conv1.weight.data.permute(1, 0, 2, 3))
        return weight


class RepRFMB(nn.Module):
    def __init__(self, in_channels, use_atten, conv=nn.Conv2d, deploy=False):
        super(RepRFMB, self).__init__()
        self.deploy = deploy
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.esa_channels = 16
        self.use_atten = use_atten
        self.c1_r = RepBlock(in_channels, self.rc, conv, use_atten, self.deploy)
        self.c2_r = RepBlock(self.remaining_channels, self.rc, conv, use_atten, self.deploy)

        self.c4 = RepBlock(self.remaining_channels, self.rc, conv, use_atten, self.deploy)
        self.act = nn.GELU()

        if use_atten:
            self.esa = ESA(self.esa_channels, in_channels)
        self.use_atten = use_atten

    def forward(self, input):
        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1)

        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2)

        r_c4 = self.act(self.c4(r_c2))

        if self.use_atten:
            out_fused = self.esa(r_c4)
        else:
            out_fused = r_c4

        return out_fused + input


class IFADNet(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_feat=36,
        num_block=6,
        num_out_ch=3,
        upscale=4,
        deploy=False,
        rgb_mean=(0.4488, 0.4371, 0.4040),
    ):
        super(IFADNet, self).__init__()

        self.deploy = deploy
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.conv = BSConvU
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding="same")

        self.B1 = RepRFMB(in_channels=num_feat, use_atten=False, conv=self.conv, deploy=self.deploy)
        self.B2 = RepRFMB(in_channels=num_feat, use_atten=True, conv=nn.Conv2d, deploy=self.deploy)
        self.B3 = RepRFMB(in_channels=num_feat, use_atten=False, conv=self.conv, deploy=self.deploy)
        self.B4 = RepRFMB(in_channels=num_feat, use_atten=True, conv=nn.Conv2d, deploy=self.deploy)
        self.B5 = RepRFMB(in_channels=num_feat, use_atten=False, conv=self.conv, deploy=self.deploy)
        self.B6 = RepRFMB(in_channels=num_feat, use_atten=True, conv=nn.Conv2d, deploy=self.deploy)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.upsampler = PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)

    def forward(self, input):
        self.mean = self.mean.type_as(input)
        input = input - self.mean

        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        
        fusion = self.GELU(self.c1(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1)))+ out_fea
        out = self.upsampler(fusion)+ self.mean

        return out
