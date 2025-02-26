import numpy as np
import torch
from monai.networks.blocks import UnetResBlock
from torch import nn
from mamba_ssm import Mamba
from torch.cuda.amp import autocast


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
    def __init__(self, g_channel, en_channel, reverse_double=True, final_decode=False):
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


if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    g_channel = 32
    en_channel = 64
    feature_map_size = (32, 32, 32)  

    g = torch.randn(batch_size, g_channel, *feature_map_size).float()
    x_en = torch.randn(batch_size, en_channel, *feature_map_size).float()

    g.to(device)
    x_en.to(device)

    skip_attention = SkipAttention(
        g_channel=g_channel,
        en_channel=en_channel,
        reverse_double=True,  # 使用双重反向处理
        final_decode=False  # 不使用解码操作
    ).to(device)

    output = skip_attention(g, x_en)

    print(f"Output shape: {output.shape}")
