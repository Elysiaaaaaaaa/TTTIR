## TTTIR
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

import warnings

from timm.models.layers import DropPath
import math
from pytorch_wavelets import DWTForward
class BandNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        return (x - mean) / (std + self.eps)

class HFSC(nn.Module):
    def __init__(self, channels, num_outputs=3, reduction=4):
        super().__init__()

        self.num_outputs = num_outputs
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')

        hidden = max(channels // reduction, 8)

        self.band_encoder = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )

        self.fusion_mlp = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, num_outputs * 4, 1)
        )

        self.band_norm = BandNorm()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels
            ),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU()
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, K, C, H, W]
        """
        B, C, H, W = x.shape
        K = self.num_outputs
        ll, yh = self.dwt(x)
        lh, hl, hh = yh[0][:, :, 0], yh[0][:, :, 1], yh[0][:, :, 2]
        ll = self.band_norm(ll)
        lh = self.band_norm(lh)
        hl = self.band_norm(hl)
        hh = self.band_norm(hh)

        ll_e = self.band_encoder(ll)
        lh_e = self.band_encoder(lh)
        hl_e = self.band_encoder(hl)
        hh_e = self.band_encoder(hh)

        feat = torch.cat([ll_e, lh_e, hl_e, hh_e], dim=1)
        logits = self.fusion_mlp(feat)
        logits = logits.view(B, K, 4, logits.shape[-2], logits.shape[-1])
        weights = F.softmax(logits, dim=2)
        bands = torch.stack([ll, lh, hl, hh], dim=1)  # [B,4,C,H/2,W/2]
        outputs = []
        for i in range(K):
            w = weights[:, i]         # [B,4,H/2,W/2]
            w = w.unsqueeze(2)        # [B,4,1,H/2,W/2]
            y = (bands * w).sum(dim=1)  # [B,C,H/2,W/2]
            y = self.refine(y)
            y = self.up(y)
            y = F.interpolate(
                y,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            outputs.append(y)
        out = torch.stack(outputs, dim=1)  # [B,K,C,H,W]
        return out
    
  
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class TTT(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, head_dim * 2, bias=qkv_bias)
        self.k = nn.Linear(dim, head_dim, bias=qkv_bias)
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        trunc_normal_(self.w3, std=.02)
        self.proj = nn.Linear(head_dim, dim)

        equivalent_head_dim = 9
        self.scale = equivalent_head_dim ** -0.5
        # The equivalent head_dim of 3x3dwc branch is 1x(3x3)=9 (1 channel, 3x3 kernel)
        # We used this equivalent_head_dim to compute self.scale in our earlier experiments
        # Using self.scale=head_dim**-0.5 (head_dim of simplified SwiGLU branch) leads to similar performance


    def inner_train_3x3dwc(self, k, v, w, lr=1, implementation='prod'):
        """
        Args:
            k (torch.Tensor): Spatial key tensor of shape [B, C, H, W]
            v (torch.Tensor): Spatial value tensor of shape [B, C, H, W]
            w (torch.Tensor): 3x3 convolution weights of shape [C, 1, 3, 3]
            lr (float, optional): Learning rate for inner-loop update. Default: 1.0
            implementation (str, optional): Implementation method, 'conv' or 'prod'. Default: 'prod'

        Returns:
            torch.Tensor: Updated convolution weights
        """
        B, C, H, W = k.shape
        e = - v / float(v.shape[2] * v.shape[3]) * self.scale
        if implementation == 'conv':
            g = F.conv2d(k.reshape(1, B * C, H, W), e.reshape(B * C, 1, H, W), padding=1, groups=B * C)
            g = g.transpose(0, 1)
        elif implementation == 'prod':
            k = F.pad(k, (1, 1, 1, 1))
            outs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys = 1 + dy
                    xs = 1 + dx
                    dot = (k[:, :, ys: ys + H, xs: xs + W] * e).sum(dim=(-2, -1))
                    outs.append(dot)
            g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        else:
            raise NotImplementedError

        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)

        if w.shape[0] == self.head_dim:
            w = w.repeat(B, 1, 1, 1) - lr * g
        else:
            w = w - lr * g
        return w

    def forward(self, x, target_state=None, w1=None, w2=None, w3=None):
        """
        Args:
            x: [B, H, W, C]
            target_state: [B, H, W, C]
        """
        target_state = target_state.detach()
        if len(x.shape) == 4:
            b, h, w, c = target_state.shape
            n = h * w
            x = x.view(b, n, c)
            target_state = target_state.view(b, n, c)
        d = c // self.num_heads
        
        # --- Prepare q/k/v ---
        q2, k2 = torch.split(
            self.qkv(x), [d, d], dim=-1
        )
        v2 = self.k(target_state)
        
        # spatial branch
        q2 = q2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        k2 = k2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        
        if w3 is None:
            w3 = self.inner_train_3x3dwc(
                k2, v2, self.w3, implementation='prod'
            )
        else:
            w3 = self.inner_train_3x3dwc(
                k2, v2, w3, implementation='prod'
            )

        x2 = F.conv2d(
            q2.reshape(1, b * d, h, w),
            w3,
            padding=1,
            groups=b * d
        )
        x = x2.reshape(b, d, n).transpose(1, 2)

        x = self.proj(x)
        x = x.view(b, h, w, c)

        return x, w3


    
      
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=2,squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),
            nn.Conv2d(num_feat//compress_ratio, num_feat // compress_ratio, 3, 1, 1,groups=num_feat//compress_ratio),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),
            nn.Conv2d(num_feat, num_feat, 3,1,padding=2,groups=num_feat,dilation=2),
            ChannelAttention(num_feat),
        )

    def forward(self, x):
        return self.cab(x)
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class HSE(nn.Module):
    def __init__(
            self,
            d_model,
            focal_level = None,
            space_Foc = None,
            freq_Foc = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.focal_level = focal_level
        self.space_foc = space_Foc
        self.freq_foc = freq_Foc
        self.TTT = TTT(dim=self.d_model, num_heads=self.d_model // 8)
        self.blocks = nn.ModuleList([])
        for _ in range(self.focal_level):
            self.blocks.append(
                PreNorm(self.d_model, FeedForward(dim=self.d_model))
            )
        self.Layernorm = nn.LayerNorm(self.d_model)
        
    def forward(self, x: torch.Tensor):
        w3 = None

        spatial_states = self.space_foc(x)  # [B,L,C,H,W]
        frequency_states = self.freq_foc(x)

        spatial_states = spatial_states.permute(0, 1, 3, 4, 2)
        frequency_states = frequency_states.permute(0, 1, 3, 4, 2)
        current_state = (spatial_states[:, 0] + frequency_states[:, 0]) / 2
        for t in range(1, frequency_states.shape[1]):
            target_state = (frequency_states[:, t] + spatial_states[:, t])/2
            x_, w3 = self.TTT(
                current_state,
                target_state=target_state,
                w3=w3
            )
            x_ = self.Layernorm(x_)
            current_state = x_ + current_state
            current_state = self.blocks[t](current_state) + current_state
        out = current_state.permute(0, 3, 1, 2)
        return out

class HSEBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            space_Foc=None,
            freq_Foc = None,
            focal_level=None,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.state_evolution = HSE(
            d_model=hidden_dim, 
            focal_level=focal_level,
            space_Foc=space_Foc,
            freq_Foc = freq_Foc,
            )
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(1, hidden_dim, 1, 1))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(1, hidden_dim, 1, 1))
        self.hidden_dim =hidden_dim

    def forward(self, input):
        x = self.ln_1(input.permute(0,2,3,1)).permute(0,3,1,2)
        state_feature = self.state_evolution(x) 
        x = x*state_feature
        x = input*self.skip_scale + self.drop_path(x)
        x = x*self.skip_scale2 + \
            self.conv_blk(
                self.ln_2(x.permute(0,2,3,1)).permute(0,3,1,2)
                )
        return x

class HSSC(nn.Module):

    def __init__(self, 
                 dim,
                 proj_drop=0., 
                 focal_level=2, 
                 focal_window=7, 
                 focal_factor=2, 
                 use_postln=False
                 ):

        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        self.f = nn.Conv2d(dim, dim+(self.focal_level), kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.focal_layers = nn.ModuleList()
        self.proj_drop = nn.Dropout(proj_drop)
        if self.use_postln:
            self.ln = nn.LayerNorm(dim)
        self.skip_scale= nn.Parameter(torch.zeros(self.focal_level,1,dim,1,1))

        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, 
                              kernel_size=kernel_size, 
                              stride=1, 
                              groups=dim, 
                              padding=kernel_size//2, 
                              bias=False),
                    nn.GELU(),
                    )
                )

    def forward(self, x):
        B, C, nH, nW = x.shape
        x = self.f(x)
        ctx, gates = torch.split(x, (C, self.focal_level), 1)
        ctx_all = []
        for l in range(self.focal_level):                     
            out = self.focal_layers[l](ctx)
            ctx = ctx*self.skip_scale[l] + self.proj_drop(out)             # 每层注意力加法
            ctx_all.append(ctx*gates[:, l:l+1])
        ctx_all.reverse()
        ctx_all = torch.stack(ctx_all, dim=1)  # dim=0 在第一个维度堆叠
        return ctx_all

class SEBlock(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 smooth=False,
                 depths=None,
                 drop_path_rate=0.1,
                 focal_level=3
                ):
        super(SEBlock, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        
        self.space_Foc = HSSC(
                dim = embed_dim,
                proj_drop=0., 
                focal_level=focal_level, 
                focal_window=3, 
                focal_factor=0, 
                use_postln=False
                )
        self.freq_Foc = HFSC(embed_dim)
        self.blocks = nn.ModuleList()
        for i in range(depths):
            self.blocks.append(HSEBlock(
                hidden_dim=embed_dim,
                drop_path=dpr[i],
                focal_level=focal_level,
                space_Foc=self.space_Foc,
                freq_Foc = self.freq_Foc,
                ))


        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x):
        x_st = x
        for blk in self.blocks:
            x = blk(x)
        
        return self.conv(x) + x_st

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
