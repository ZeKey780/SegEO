import pytest
import torch

from skip_mamba_attention.mix_attention3D import MambaLayer, BasicConvBlock, ChannelAttention, SkipAttention

# ȷ���豸
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestMambaLayer:
    def test_forward_channel(self):
        layer = MambaLayer(dim=64, channel=True).to(device)
        x = torch.randn(2, 16, 4, 4, 4, device=device)  # B, d_model, H, W, D
        output = layer(x)
        assert output.shape == (2, 16, 4, 4, 4), "�����״����ȷ"

    def test_forward_patch(self):
        layer = MambaLayer(dim=16, channel=False).to(device)
        x = torch.randn(2, 16, 4, 4, 4, device=device)  # B, d_model, H, W, D
        output = layer(x)
        assert output.shape == (2, 16, 4, 4, 4), "�����״����ȷ"

    def test_invalid_input_shape(self):
        layer = MambaLayer(dim=16).to(device)
        x = torch.randn(2, 15, 4, 4, 4, device=device)  # d_model ��ƥ��
        with pytest.raises(AssertionError):
            layer(x)


class TestBasicConvBlock:
    def test_forward(self):
        block = BasicConvBlock(input_channels=3, output_channels=16, padding=1,stride=2).to(device)
        x = torch.randn(2, 3, 10, 10, 10, device=device)  # B, C, D, H, W
        output = block(x)
        print("输出维度：", output.shape)
        # assert output.shape == (2, 16, 10, 10, 10), "�����״����ȷ"

    def test_invalid_input_channel(self):
        block = BasicConvBlock(input_channels=3, output_channels=16).to(device)
        x = torch.randn(2, 4, 10, 10, 10, device=device)  # ����ͨ����ƥ��
        with pytest.raises(AssertionError):
            block(x)


class TestChannelAttention:
    def test_forward(self):
        attention = ChannelAttention(g_c=3, x_c=16, feature_map_size=(10, 10, 10)).to(device)
        g = torch.randn(2, 3, 10, 10, 10, device=device)
        x = torch.randn(2, 16, 10, 10, 10, device=device)
        output = attention(g, x)
        assert output.shape == (2, 16, 10, 10, 10), "�����״����ȷ"

    def test_invalid_input_shape(self):
        attention = ChannelAttention(g_c=3, x_c=16, feature_map_size=(10, 10, 10)).to(device)
        g = torch.randn(2, 4, 10, 10, 10, device=device)  # C_g ��ƥ��
        x = torch.randn(2, 16, 10, 10, 10, device=device)
        with pytest.raises(AssertionError):
            attention(g, x)


class TestSkipAttention:
    def test_forward(self):
        attention = SkipAttention(g_channel=3, en_channel=16, feature_map_size=(10, 10, 10)).to(device)
        g = torch.randn(2, 3, 10, 10, 10, device=device)
        x = torch.randn(2, 16, 10, 10, 10, device=device)
        output = attention(g, x)
        assert output.shape == (2, 16, 10, 10, 10), "�����״����ȷ"

    def test_invalid_channel_x(self):
        attention = SkipAttention(g_channel=3, en_channel=16, feature_map_size=(10, 10, 10)).to(device)
        x = torch.randn(2, 15, 10, 10, 10, device=device)  # C_x ��ƥ��
        g = torch.randn(2, 3, 10, 10, 10, device=device)
        with pytest.raises(AssertionError):
            attention(g, x)


# ���в���
if __name__ == "__main__":
    pytest.main()
