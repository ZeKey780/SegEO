import torch
from torch import nn


from pool_interp import InterPool
from upsample_space3D import DySample3D


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


if __name__ == '__main__':
    # 设置参数
    final_channels = 4
    conv_input_channel = 8
    g_channel = 4
    sum_channel = True
    mix_up = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模块
    upsample = UpSampling(
        final_channels_in=final_channels,
        conv_in_channel=conv_input_channel,
        g_channel=g_channel,
        sum_c=sum_channel,
        mix_up=mix_up
    ).to(device)

    # 定义输入数据
    batch_size = 2
    depth, height, width = 8, 16, 16
    x = torch.randn(batch_size, conv_input_channel, depth, height, width).to(device)
    g = torch.randn(batch_size, g_channel, depth, height, width).to(device)

    # 前向传播
    output = upsample(x, g)

    # 打印结果
    print("Input x shape:", x.shape)
    print("Input g shape:", g.shape)
    print("Output shape:", output.shape)
