import pytest
import torch

from upsample.upsample_space3D import DySample3D


def test_dysample3d_initialization():
    # 测试初始化
    model = DySample3D(final_channels=64, scale=2, style='lp', groups=4)
    assert model.scale == 2
    assert model.style == 'lp'
    assert model.groups == 4
    assert model.offset.weight.shape == (3 * model.groups, 64, 1, 1, 1)

    # 测试 style 参数不合法
    with pytest.raises(AssertionError):
        DySample3D(final_channels=64, scale=2, style='invalid', groups=4)

    # 测试 style 为 'pl' 时，输入通道数不足
    with pytest.raises(AssertionError):
        DySample3D(final_channels=8, scale=2, style='pl', groups=4)

def test_init_pos_shape():
    model = DySample3D(final_channels=64)
    init_pos = model.init_pos
    expected_shape = (1, model.groups * 27, 1, 1, 1)  # scale=2, 3D网格
    assert init_pos.shape == expected_shape

def test_pixelshuffle():
    model = DySample3D(final_channels=64)
    x = torch.rand(2, 64, 7, 8, 9)
    reshuffled = model.pixelshuffle(x)
    assert reshuffled.shape == (2, 64 // 8, 14, 16, 18)  # scale=2

    # 测试输入通道数不符合
    with pytest.raises(ValueError):
        model.pixelshuffle(torch.rand(2, 63, 7, 8, 9))

def test_sample():
    model = DySample3D(final_channels=64)
    x = torch.rand(2, 64, 7, 8, 9)
    offset = torch.rand(2, 3, 1, 7, 8, 9)
    sampled = model.sample(x, offset)
    assert sampled.shape == (2, 3 * model.groups, 14, 16, 18)  # scale=2

def test_forward():
    model = DySample3D(final_channels=64)
    x = torch.rand(2, 64, 7, 8, 9)
    output = model(x)
    assert output.shape == (2, 3 * model.groups, 14, 16, 18)  # scale=2

    # 测试输入形状不匹配
    with pytest.raises(RuntimeError):
        model(torch.rand(2, 32, 7, 8, 9))  # 通道数不足

if __name__ == "__main__":
    pytest.main()