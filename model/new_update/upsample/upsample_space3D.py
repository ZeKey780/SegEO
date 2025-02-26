import torch
import torch.nn as nn
import torch.nn.functional as F



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
        # self.conv = BasicConvBlock(input_channels=in_channels, output_channels=in_channels / 2)
        self.conv = nn.Conv3d(conv_input_channel, final_channels, kernel_size=1, stride=1, padding=0)
        normal_init(self.conv, std=0.01)

    def _init_pos(self):
        # Create 3D grid, normalize to [-1, 1] range
        d = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        w = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale

        grid = torch.stack(torch.meshgrid([d, h, w], indexing='ij'))

        grid = grid.unsqueeze(0)

        grid = grid.repeat(1, self.groups, 1, 1, 1)

        return grid.reshape(1, -1, 1, 1, 1)

    def pixelshuffle(self, x: torch.Tensor) -> torch.Tensor:

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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.rand(2, 48, 7, 8, 9).to(device)  # 3D input with shape (batch, channels, depth, height, width)
    dys = DySample3D(24, 48).to(device)
    print(dys(x).shape)
