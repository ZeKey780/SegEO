import numpy as np


def trilinear_interpolation_3d(input_data):
    # 获取输入数据的批量大小、通道数和深度、高度、宽度
    B, C, H, W, D = input_data.shape

    # 创建输出图像，尺寸是原图的两倍
    output_data = np.zeros((B, C, 2 * H, 2 * W, 2 * D), dtype=input_data.dtype)

    # 对每个批次和每个通道进行插值
    for b in range(B):
        for c in range(C):
            # 获取单通道的 3D 图像
            input_image = input_data[b, c]

            # 对每个坐标进行插值
            for z in range(2 * D):
                for y in range(2 * H):
                    for x in range(2 * W):
                        # 计算输入图像中的浮点位置
                        z_in = z / 2.0
                        y_in = y / 2.0
                        x_in = x / 2.0

                        # 获取相邻的体素索引
                        z0, y0, x0 = int(np.floor(z_in)), int(np.floor(y_in)), int(np.floor(x_in))
                        z1, y1, x1 = min(z0 + 1, D - 1), min(y0 + 1, H - 1), min(x0 + 1, W - 1)

                        # 计算插值权重
                        dz, dy, dx = z_in - z0, y_in - y0, x_in - x0

                        # 进行三线性插值，沿 x、y 和 z 方向分别插值
                        c00 = input_image[y0, x0, z0] * (1 - dx) + input_image[y0, x1, z0] * dx
                        c01 = input_image[y0, x0, z1] * (1 - dx) + input_image[y0, x1, z1] * dx
                        c10 = input_image[y1, x0, z0] * (1 - dx) + input_image[y1, x1, z0] * dx
                        c11 = input_image[y1, x0, z1] * (1 - dx) + input_image[y1, x1, z1] * dx

                        c0 = c00 * (1 - dy) + c01 * dy
                        c1 = c10 * (1 - dy) + c11 * dy

                        # 最终在z方向上进行插值
                        output_data[b, c, y, x, z] = c0 * (1 - dz) + c1 * dz

    return output_data


# 测试
input_data = np.random.rand(2, 3, 4, 4, 4)  # 示例输入数据，形状为 (B, C, H, W, D)
output_data = trilinear_interpolation_3d(input_data)
print("Input shape:", input_data.shape)
print("Output shape:", output_data.shape)
