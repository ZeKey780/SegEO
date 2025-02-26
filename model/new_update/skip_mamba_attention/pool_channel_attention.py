import torch
from torch import nn
from mamba_ssm import Mamba


class SkipAttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(SkipAttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        self.avg_x = nn.AdaptiveAvgPool3d(1)
        self.avg_g = nn.AdaptiveAvgPool3d(1)

        self.mam_x_g = Mamba(
            d_model=2
        )
        # self.mam_g = Mamba(d_model=1)
        self.L_x = nn.Linear(F_l, F_l)
        self.L_g = nn.Linear(F_g, F_l)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)

        bx, cx, dx, hx, wx = x.size()
        bg, cg, dg, hg, wg = g.size()

        avg_pool_x = self.avg_x(x).view(bx, cx, 1)
        avg_pool_g = self.avg_g(g).view(bg, cg, 1)

        mam_x = self.mam_x(avg_pool_x)
        # mam_g = self.mam_g(avg_pool_g)

        # 输出avg_pool_x和avg_pool_g特征的维度
        print('X', avg_pool_x.size())
        print("g", avg_pool_g.size())

        print('X', mam_x.size())
        # print("g", mam_g.size())

        # channel_att_x = self.L_x(avg_pool_x)
        # channel_att_g = self.L_g(avg_pool_g)
        channel_att = mam_x
        scale = torch.sigmoid(channel_att).view(bx, cx, 1, 1, 1)
        #
        # psi = self.psi(psi)
        # psi = psi * scale

        return x * scale


if __name__ == '__main__':
    model = SkipAttentionBlock(128, 128, 32)
    g = torch.randn(2, 128, 16, 16, 16)
    x = torch.randn(2, 128, 16, 16, 16)
    x1 = x.flatten(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    g = g.to(device)
    x = x.to(device)

    output = model(g, x)

    print(x1.size())

    print(output.size())
