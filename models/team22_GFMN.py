import torch
import torch.nn as nn
import torch.nn.functional as F



# Layer Norm
class LayerNorm(nn.Module):
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

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class FeedForward2(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FeedForward2, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        # self.dw5conv = nn.Conv2d(hidden_features*2, hidden_features*2, 5, 1, 5//2, groups= hidden_features*2) 

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.act = nn.GELU()
        self.decompose = nn.Conv2d(
            in_channels=hidden_features,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            hidden_features, init_value=1e-5, requires_grad=True)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x = F.gelu(x1) * x2
        x = x1 * x2

        x = x + self.sigma(x - self.act(self.decompose(x)))

        x = self.project_out(x)
        return x

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


    
class SAFM7(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Multiscale feature representation
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # Feature aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU() #nn.ReLU() #nn.SiLU(inplace=True) #

        self.proj_1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1)
        
        self.sigma = ElementScale(
            dim, init_value=1e-5, requires_grad=True)
        

    def forward(self, x):
        h, w = x.size()[-2:]
        x = self.proj_1(x)
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act(x)


        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**(i+1), w//2**(i+1))
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        # Feature modulation
        out = self.act(out) * x
        return out

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) #GRN(dim)
        self.norm2 = LayerNorm(dim) #GRN(dim)

        self.safm = SAFM7(dim) 
        self.ffd = FeedForward2(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ffd(self.norm2(x)) + x
        return x

class GFMN(nn.Module):
    def __init__(self, dim=32, n_blocks=8, ffn_scale=1.8, upscaling_factor=4):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

        # self.norm = GRN(dim)
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

    def forward(self, x):

        ident = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean)
        x = self.to_feat(x)
        # x = self.norm(x)
        x = self.feats(x)
        x = self.to_img(x)
        x = x  + self.mean
        return x + ident

