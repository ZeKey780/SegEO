
import torch.nn.functional as F
import torch
from monai.networks.blocks import UnetResBlock
from torch import nn
from mamba_ssm import Mamba
from torch.cuda.amp import autocast


def normal_init(module, mean=0, std=1.0, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample3D(nn.Module):
    def __init__(self, final_channels, conv_input_channel, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert final_channels >= scale ** 3 and final_channels % scale ** 3 == 0
        assert final_channels >= groups and final_channels % groups == 0

        if style == 'pl':
            final_channels = final_channels // scale ** 3
            out_channels = 3 * groups
        else:
            out_channels = 3 * groups * scale ** 3

        self.offset = nn.Conv3d(final_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)

        if dyscope:
            self.scope = nn.Conv3d(final_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())
        self.conv = nn.Conv3d(conv_input_channel, final_channels, kernel_size=1, stride=1, padding=0)
        normal_init(self.conv, std=0.01)

    def _init_pos(self):

        d = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        w = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale


        grid = torch.stack(torch.meshgrid([d, h, w], indexing='ij'))
        grid = grid.unsqueeze(0)
        grid = grid.repeat(1, self.groups, 1, 1, 1)

        return grid.reshape(1, -1, 1, 1, 1)

    def pixelshuffle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pixel shuffle to the tensor `x`, moving pixels from the channel dimension to spatial dimensions.

        See: Shi et al., 2016, "Real-Time Single Image and Video Super-Resolution
        Using a nEfficient Sub-Pixel Convolutional Neural Network."

        See: Aitken et al., 2017, "Checkerboard artifact free sub-pixel convolution".

        Args:
            x: Input tensor

        Returns:
            Reshuffled version of `x`.

        Raises:
            ValueError: When spatial dimensions and coordinate dimensions do not match
            ValueError: When input channels of `x` are not divisible by (scale_factor ** spatial_dims)
        """
        dim, factor = 3, self.scale
        input_size = list(x.size())
        keeped_dim = input_size[:-(dim + 1)]
        channels = input_size[-(dim + 1)]

        if len(keeped_dim) == 2 and dim != keeped_dim[1]:
            raise ValueError(
                f"The data has a dimension of {dim}, while the coordinate dimension is {keeped_dim[1]}."
            )
        scale_divisor = factor ** dim
        if channels % scale_divisor != 0:
            raise ValueError(
                f"Number of input channels ({channels}) must be evenly "
                f"divisible by scale_factor ** dimensions ({factor}**{dim}={scale_divisor})."
            )

        spatial_start_idx = len(keeped_dim) + 1
        org_channels = int(channels // scale_divisor)
        output_size = keeped_dim + [org_channels] + [d * factor for d in input_size[spatial_start_idx:]]

        indices = list(range(spatial_start_idx, spatial_start_idx + 2 * dim))
        indices = indices[dim:] + indices[:dim]
        permute_indices = list(range(spatial_start_idx))
        for idx in range(dim):
            permute_indices.extend(indices[idx::dim])

        x = x.reshape(keeped_dim + [org_channels] + [factor] * dim + input_size[spatial_start_idx:])
        x = x.permute(permute_indices).reshape(output_size)
        return x

    def sample(self, x, offset):
        B, _, D, H, W = offset.shape
        offset = offset.view(B, 3, -1, D, H, W)

        coords_d = torch.arange(D) + 0.5
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))
        coords = coords.unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        coords = coords.view(1, 3, 1, D, H, W)

        normalizer = torch.tensor([D, H, W], dtype=x.dtype, device=x.device).view(1, 3, 1, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = coords.view(B, -1, D, H, W)

        coords = self.pixelshuffle(coords).view(
            B, 3, -1, self.scale * D, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 5, 1).contiguous().flatten(0,
                                                                                                                     1)

        return F.grid_sample(x.reshape(B * self.groups, -1, D, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * D, self.scale * H,
                                                                              self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        x = self.conv(x)
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
class InterPool(nn.Module):
    def __init__(self, g_channel, upsp_final_channel, sum_channel=True):
        super().__init__()

        self.sum_channel = sum_channel
        self.upsp_final_channel = upsp_final_channel
        self.conv_g = BasicConvBlock(g_channel, upsp_final_channel)
        self.conv_x = BasicConvBlock(upsp_final_channel, upsp_final_channel)

        self.avg_x = nn.AdaptiveAvgPool3d(1)
        self.avg_g = nn.AdaptiveAvgPool3d(1)

        self.mamba = MambaLayer(dim=2, channel=True)

        self.conv = BasicConvBlock(upsp_final_channel, upsp_final_channel, stride=2)

    def forward(self, x_upsp, g):
        bx = x_upsp.shape[0]

        g = self.conv_g(g)
        x = self.conv_x(x_upsp)

        g_pool = self.avg_g(g).view(bx, self.upsp_final_channel, 1)
        x_pool = self.avg_x(x).view(bx, self.upsp_final_channel, 1)

        connect = torch.cat((g_pool, x_pool), dim=-1)
        mamba_out = self.mamba(connect)
        if self.sum_channel:
            summed_result = mamba_out.sum(dim=-1, keepdim=True)
        else:
            summed_result = mamba_out[..., 1]

        scale = torch.sigmoid(summed_result).view(bx, self.upsp_final_channel, 1, 1, 1)
        out_scale = x_upsp * scale
        return out_scale

class UpSampling(nn.Module):
    def __init__(self, final_channels_in, conv_in_channel, g_channel, sum_c=True, mix_up=True):
        super().__init__()

        self.space_up = DySample3D(final_channels=final_channels_in, conv_input_channel=conv_in_channel)

        self.pool_inter_up = InterPool(g_channel=g_channel, upsp_final_channel=final_channels_in, sum_channel=sum_c)
        self.mix_up = mix_up
        self.raw_param = nn.Parameter(torch.randn(1))

    def forward(self, x, g):
        r = torch.sigmoid(self.raw_param)

        x_space = self.space_up(x)

        x_inter_up = self.pool_inter_up(x_space, g)

        out_up = r * x_space + (1 - r) * x_inter_up

        if self.mix_up:
            return out_up
        else:
            return x_space


class SubPixelConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2, kernel_size=3, padding=1):
        super(SubPixelConv3D, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv3d(in_channels, out_channels * (upscale_factor ** 3),
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = rearrange(x, 'b (c r1 r2 r3) d h w -> b c (d r1) (h r2) (w r3)',
                      r1=self.upscale_factor, r2=self.upscale_factor, r3=self.upscale_factor)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel=True, reverse=False):
        super().__init__()
        # print(f"MambaLayer: dim: {dim}")
        self.reverse = reverse
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type = "v1"
        )
        self.channel = channel

    def forward_patch(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)

        # 判断是否对特征图进行反向处理，来弥补mamba只能看到该向量之前的向量的问题
        if self.reverse:
            x_flat = torch.flip(x_flat, dims=[1])

        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        if self.reverse:
            x_mamba = torch.flip(x_mamba, dims=[1])
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out

    def forward_channel(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        # print("转换前：",x_flat.shape)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"

        if self.reverse:
            x_flat = torch.flip(x_flat, dims=[1])
            # print("装换后：", x_flat.shape)

        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        # 在最后恢复原来形状时，要将翻转的维度还原
        if self.reverse:
            x_mamba = torch.flip(x_mamba, dims=[1])
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel:
            out = self.forward_channel(x)
        else:
            out = self.forward_patch(x)

        return out


# 对输入特征进行正反处理，使每一个向量都能包含它的前面和后面的向量的信息
class DoubleMambaLager(nn.Module):
    def __init__(self, dim, channel=True):
        super().__init__()
        self.mamba_order = MambaLayer(dim=dim, channel=channel)

        self.mamba_order_reverse = MambaLayer(dim=dim, channel=channel, reverse=True)

    def forward(self, x):
        x_oder = self.mamba_order(x)
        x_reverse = self.mamba_order_reverse(x)

        x_out = (x_oder + x_reverse) / 2.0

        return x_out


class BasicConvBlock(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            nonlin=nn.LeakyReLU,
    ):
        super().__init__()
        self.input_channel = input_channels

        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, eps=1e-5, affine=True)
        self.act1 = nonlin(inplace=True)

    def forward(self, x):
        channel_x = x.shape[1]
        # assert channel_x == self.input_channel

        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        return y


class AttentionGate3D(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class NoAttention(nn.Module):
    def __init__(self, dec_in_channels, skip_channels, out_channels,
                 upscale_factor=2, kernel_size=3, padding=1):
        super(UpConcat3D, self).__init__()
        self.subpixel = SubPixelConv3D(dec_in_channels, out_channels, upscale_factor, kernel_size, padding)

        self.fuse_conv = nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=1)

    def forward(self, dec_feat, skip_feat):
        dec_up = self.subpixel(dec_feat)
        if dec_up.shape[2:] != skip_feat.shape[2:]:
            skip_feat = F.interpolate(skip_feat, size=dec_up.shape[2:], mode='trilinear', align_corners=False)

        fused = torch.cat((dec_up, skip_feat), dim=1)
        fused = self.fuse_conv(fused)
        return fused

class ChannelAttention(nn.Module):
    def __init__(self, g_c, x_c, final_decode=False):
        super(ChannelAttention, self).__init__()
        self.final_c = x_c

        self.final_decode = final_decode

        self.final_decode_conv = UnetResBlock(
            3,
            g_c,
            x_c,
            kernel_size=3,
            stride=1,
            norm_name="instance",
        )

        self.Conv_g = BasicConvBlock(g_c, x_c)

        self.Conv_x = BasicConvBlock(x_c, x_c)

        self.avg_x = nn.AdaptiveAvgPool3d(1)
        self.avg_g = nn.AdaptiveAvgPool3d(1)

        self.maChannel = MambaLayer(
            dim=2
        )


    def forward(self, g, x_en):
        bx = x_en.shape[0]

        gc1 = self.Conv_g(x_en)
        xc1 = self.Conv_x(g)

        g_pool = self.avg_g(gc1).view(bx, self.final_c, 1)
        x_pool = self.avg_x(xc1).view(bx, self.final_c, 1)

        connect = torch.cat((g_pool, x_pool), dim=-1)
        mamba_out = self.maChannel(connect)

        summed_result = mamba_out.sum(dim=-1, keepdim=True)

        scale = torch.sigmoid(summed_result).view(bx, self.final_c, 1, 1, 1)

        # channel_att = (xmc + gmc) / 2.0
        # scale = torch.sigmoid(channel_att)
        #
        # if self.final_decode:
        #     x_en = self.final_decode_conv(x_en)

        return scale


class SkipAttention(nn.Module):
    def __init__(self, g_channel, en_channel, reverse_double=False, final_decode=False):
        super(SkipAttention, self).__init__()
        # self.C_x = x_channel
        self.final_decode = final_decode

        self.final_decode_conv = UnetResBlock(
            3,
            g_channel,
            en_channel,
            kernel_size=3,
            stride=1,
            norm_name="instance",
        )

        self.Conv_g = BasicConvBlock(g_channel, en_channel)

        self.Conv_x = BasicConvBlock(en_channel, en_channel)
        if reverse_double:
            self.maSp_x = DoubleMambaLager(dim=en_channel, channel=False)
            self.masP_g = DoubleMambaLager(dim=en_channel, channel=False)
        else:
            self.maSp_x = MambaLayer(

                dim=en_channel,
                channel=False
            )
            self.masP_g = MambaLayer(
                dim=en_channel,
                channel=False
            )

        self.channel_att = ChannelAttention(g_channel, en_channel)

    def forward(self, g, x_en):
        channel_scale = self.channel_att(g, x_en)

        gc1 = self.Conv_g(x_en)
        xc1 = self.Conv_x(g)

        xmsp = self.maSp_x(xc1)
        gmsp = self.masP_g(gc1)

        space_attention = (xmsp + gmsp) / 2.0
        scale = torch.sigmoid(space_attention)

        if self.final_decode:
            x_en = self.final_decode_conv(x_en)

        scale = scale * channel_scale

        attention_out = x_en * scale

        # out = (channel_attention_out + space_attention_out) / 2.0

        return attention_out
class UpWithAttention(nn.Module):
    def __init__(self, final_channels, conv_in_channel, g_channel, out_channel, reverse_double=False,
                 sum_c=True,
                 final_decode=False,
                 mix_up=True, norm_name="instance", dims=3):
        super().__init__()

        self.upSampling = UpSampling(final_channels_in=final_channels, conv_in_channel=conv_in_channel,
                                     g_channel=g_channel, sum_c=sum_c, mix_up=mix_up)

        self.attention = SkipAttention(g_channel=g_channel, en_channel=final_channels,
                                    reverse_double=reverse_double, final_decode=final_decode)

        self.conv = UnetResBlock(
            dims,
            final_channels + final_channels,
            out_channel,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, x, skip):
        up_x = self.upSampling(x, skip)

        attention_skip = self.attention(up_x, skip)

        connect = torch.cat((up_x, attention_skip), dim=1)
        out = self.conv(connect)
        return out


if __name__ == '__main__':
    final_channels = 48
    conv_in_channel = 48
    g_channel = 48
    out_channel = 48
    dims = 3
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model = UpWithAttention(
        final_channels=final_channels,
        conv_in_channel=conv_in_channel,
        g_channel=g_channel,
        out_channel=out_channel,
        norm_name="instance",
        dims=dims,
    ).to(device)


    x = torch.randn(2, conv_in_channel, 64, 64, 64).to(device)
    skip = torch.randn(2, g_channel, 128, 128, 128).to(device)


    output = model(x, skip)

    print("Output shape:", output.shape)
