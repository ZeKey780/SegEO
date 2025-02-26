import torch
import torch.nn.functional as F
from torch import nn

from skip_mamba_attention.mix_attention3D import BasicConvBlock


def global_average_pooling_3d(input_data):
    # 对输入数据进行3D全局平均池化，得到形状为 (B, C, 1, 1, 1) 的特征图
    _, _, H, W, D = input_data.shape
    pooled_tensor = F.avg_pool3d(input_data, kernel_size=(H, W, D))
    return pooled_tensor


def trilinear_upsample(input_data, scale_factor=2):
    # 进行三线性插值上采样
    return F.interpolate(input_data, scale_factor=scale_factor, mode='trilinear', align_corners=False)


def combine_features(input_data):
    # 对输入进行全局池化
    pooled_feature_map = global_average_pooling_3d(input_data)

    # 对输入进行上采样
    upsampled_feature_map = trilinear_upsample(input_data)

    # 逐元素相乘，池化后的特征图会广播到上采样特征图的空间维度
    combined_feature_map = upsampled_feature_map * pooled_feature_map
    return combined_feature_map

class up(nn.Moudle):
    def __init__(self, x_c, xen_c):
        super().__init__()

        self.xen_conv = BasicConvBlock(xen_c, xen_c, padding=1, stride=2)


        self.average_pooling = nn.AdaptiveAvgPool3d(1);

# 示例使用
input_data = torch.rand(2, 3, 8, 8, 8)  # (B, C, H, W, D)
output_data = combine_features(input_data)

print("输入数据的形状:", input_data.shape)
print("输出数据的形状（经过上采样和相乘后）:", output_data.shape)
