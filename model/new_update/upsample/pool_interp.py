import torch
from torch import nn


from mix_attention3D import BasicConvBlock, MambaLayer


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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    x_upsp = torch.rand(2, 48, 8, 8, 9).to(device)
    g = torch.rand(2, 24, 7, 8, 10).to(device)


    model = InterPool(g_channel=24, upsp_final_channel=48, sum_channel=True).to(device)

    # 前向传递
    output = model(x_upsp, g)

    # 打印输入和输出的形状
    print(f"Input x_upsp shape: {x_upsp.shape}")
    print(f"Input g shape: {g.shape}")
    print(f"Output shape: {output.shape}")
