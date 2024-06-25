from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_local_weights(residual, ksize, padding):
    pad = padding
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
    unfolded_residual = residual_pad.unfold(2, ksize, 3).unfold(3, ksize, 3)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight


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
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
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
        self.dw1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,1),
            stride=stride,
            padding=(padding,0),
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode='reflect',
        )
        self.dw2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1,kernel_size),
            stride=stride,
            padding=(0,padding),
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode='reflect',
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw2(self.dw1(fea))
        return fea


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

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
    def __init__(self, c_dim, reduction):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c_dim, c_dim, (1, 1), padding='same'),
                                  nn.GELU(),
                                  CCALayer(c_dim, reduction),
                                  nn.Conv2d(c_dim, c_dim, (3, 3), padding='same', groups=c_dim))

    def forward(self, x):
        ca_x = self.body(x)
        ca_x += x
        return ca_x


class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = num_feat // 4
        self.ln=LayerNorm(num_feat)
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
        x= self.ln(input)
        c1_ = self.conv1(x)  # channel squeeze
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

        return x * m


class MDSA(nn.Module):
    def __init__(self, c_dim, conv):
        super().__init__()
        self.body = nn.Sequential(ESA(c_dim, conv))

    def forward(self, x):
        sa_x = self.body(x)
        sa_x += x
        return sa_x


class EADB(nn.Module):
    def __init__(self, in_channels, conv=nn.Conv2d):
        super(EADB, self).__init__()
        kwargs = {'padding': 1}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3, **kwargs)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.esa = MDSA(in_channels, conv)
        self.cca = ECCA(in_channels, reduction=16)

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

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        out_fused = self.esa(out)  # MDSA
        out_fused = self.cca(out_fused)  # ECCA
        return out_fused + input



class MSN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=28, num_block=8, num_out_ch=3, upscale=4,
                 rgb_mean=(0.4488, 0.4371, 0.4040), p=0.25):
        super(MSN, self).__init__()
        kwargs = {'padding': 1}
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.conv = BSConvU
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding='same')

        self.B1 = EADB(in_channels=num_feat, conv=self.conv)
        self.B2 = EADB(in_channels=num_feat, conv=self.conv)
        self.B3 = EADB(in_channels=num_feat, conv=self.conv)
        self.B4 = EADB(in_channels=num_feat, conv=self.conv)
        self.B5 = EADB(in_channels=num_feat, conv=self.conv)
        self.B6 = EADB(in_channels=num_feat, conv=self.conv)
        self.B7 = EADB(in_channels=num_feat, conv=self.conv)
        self.B8 = EADB(in_channels=num_feat, conv=self.conv)
        # self.B9 = EADB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        # self.B10 = EADB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Conv2d(num_feat, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)

        # self.to_RGB = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.upsampler = PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)

    def forward(self, input):
        self.mean = self.mean.type_as(input)
        input = input - self.mean
        # denosed_input = denosed_input - self.mean
        # SR
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        out = self.upsampler(self.c2(self.GELU(self.c1(out_B1+ out_B2+ out_B3+ out_B4+
                                                                  out_B5+ out_B6+ out_B7+ out_B8))) + out_fea) + self.mean
        # del out_B1, out_B2, out_B3, out_B4, out_B5
        # # Denose
        # out_fea = self.fea_conv(denosed_input)
        # out_B1 = self.B1(out_fea)
        # out_B2 = self.B2(out_B1)
        # out_B3 = self.B3(out_B2)
        # out_B4 = self.B4(out_B3)
        # out_B5 = self.B5(out_B4)
        # # out_B6 = self.B6(out_B5)
        # # out_B7 = self.B7(out_B6)
        # # out_B8 = self.B8(out_B7)
        # out = self.c2(self.GELU(self.c1(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5], dim=1)))) + out_fea
        # if detach_ture:
        #     output_denosed = self.to_RGB(out.detach()) + self.mean
        # else:
        #     output_denosed = self.to_RGB(out) + self.mean
        return out

if __name__ == '__main__':
    window_size = 8
    upscale = 4
    height = (2040 // upscale)
    width = (1340 // upscale)
    model = MSN()
    # img = plt.imread(r'C:\Users\Bolt\Desktop\Set5\LRbicx2\butterfly.png')
    # img_tensor = torch.from_numpy(img)
    # img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    # out1 = model(img_tensor)
    # plt.figure(0)
    # plt.imshow(out1.squeeze(0).permute(1, 2, 0).detach().numpy())

    # print(model)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    x = torch.randn((1, 3, height, width))
    # a = time.time()
    # x = model(x)
    # b = time.time()
    # print(x.shape)
    # print(b-a)

    iterations = 100  # 重复计算的轮次

    device = torch.device("cuda:0")
    model.to(device)

    random_input = x.to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(200):
        with torch.no_grad():
            _= model(random_input)

    # 测速
    times = torch.zeros(iterations)  # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _= model(random_input)
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
    # from utility.model_summary import get_model_flops, get_model_activation
    # from thop import profile

    input_dim = (3, 320, 180)  # set the input dimension
    # activations, num_conv = get_model_activation(model, input_dim)
    # activations = activations / 10 ** 6
    # print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    # print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    # flops = get_model_flops(model, input_dim, False)
    # flops = flops / 10 ** 9
    # print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))

    # input = torch.randn((1, 3, 320, 180)).cuda()
    # x, y = profile(model, inputs=(input,))
    # print("{:>16s} : {:<.4f} [G]".format("FLOPs", x / 10 ** 9))
# print(torch.cuda.memory_summary())