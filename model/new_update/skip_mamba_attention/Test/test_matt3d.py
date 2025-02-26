import pytest
import torch

from skip_mamba_attention.mix_attention3D import MambaLayer, BasicConvBlock, ChannelAttention, SkipAttention


# 测试 MambaLayer
def test_mamba_layer_forward_channel():
    layer = MambaLayer(dim=16, channel=True)
    x = torch.rand(2, 16, 4, 4, dtype=torch.float32)  # 正常输入

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在模型定义前确保模型也在设备上
    layer.to(device)
    # 在将数据传入模型前确保数据也在设备上
    x = x.to(device)

    output = layer(x)
    assert output.shape == (2, 16, 4, 4), "输出形状不正确"


def test_mamba_layer_forward_patch():
    layer = MambaLayer(dim=16, channel=False)
    x = torch.rand(2, 16, 4, 4, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在模型定义前确保模型也在设备上
    layer.to(device)
    # 在将数据传入模型前确保数据也在设备上
    x = x.to(device)

    output = layer(x)
    assert output.shape == (2, 16, 4, 4), "输出形状不正确"


def test_mamba_layer_invalid_input():
    layer = MambaLayer(dim=16)
    x = torch.rand(2, 15, 5, 5, dtype=torch.float32)  # 错误的输入维度

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在模型定义前确保模型也在设备上
    layer.to(device)
    # 在将数据传入模型前确保数据也在设备上
    x = x.to(device)

    with pytest.raises(AssertionError):
        layer(x)


# 测试 BasicConvBlock
def test_basic_conv_block():
    block = BasicConvBlock(input_channels=3, output_channels=16)
    x = torch.rand(2, 3, 8, 8, 8)  # 输入形状 [B, C, D, H, W]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在模型定义前确保模型也在设备上
    block.to(device)
    # 在将数据传入模型前确保数据也在设备上
    x = x.to(device)

    output = block(x)
    assert output.shape == (2, 16, 8, 8, 8), "输出形状不正确"


# 测试 ChannelAttention
def test_channel_attention():
    attention = ChannelAttention(g_c=3, x_c=16, feature_map_size=(4, 4, 4))
    g = torch.rand(2, 3, 4, 4, 4)
    x = torch.rand(2, 16, 4, 4, 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在模型定义前确保模型也在设备上
    attention.to(device)
    # 在将数据传入模型前确保数据也在设备上
    g = g.to(device)
    x = x.to(device)

    output = attention(g, x)
    assert output.shape == (2, 16, 4, 4, 4), "输出形状不正确"


# 测试 SkipAttention
def test_skip_attention():
    attention = SkipAttention(g_channel=3, en_channel=16, feature_map_size=(4, 4, 4))
    g = torch.rand(2, 3, 4, 4, 4)
    x = torch.rand(2, 16, 4, 4, 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在模型定义前确保模型也在设备上
    attention.to(device)
    # 在将数据传入模型前确保数据也在设备上
    g = g.to(device)
    x = x.to(device)

    output = attention(g, x)
    assert output.shape == (2, 16, 4, 4, 4), "输出形状不正确"


# 测试 SkipAttention 的负面情况
def test_skip_attention_invalid_input():
    attention = SkipAttention(g_channel=3, en_channel=16, feature_map_size=(4, 4, 4))
    g = torch.rand(2, 3, 4, 4, 4)
    x = torch.rand(2, 15, 4, 4, 4)  # 错误的输入维度

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在模型定义前确保模型也在设备上
    attention.to(device)
    # 在将数据传入模型前确保数据也在设备上
    g = g.to(device)
    x = x.to(device)

    with pytest.raises(AssertionError):
        attention(g, x)


if __name__ == "__main__":
    pytest.main()
