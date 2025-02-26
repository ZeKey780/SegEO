import torch
import torch.nn.functional as F


def global_average_pooling_3d(input_data):
    # Ensure input_data is a tensor and has the correct type
    if not isinstance(input_data, torch.Tensor):
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
    else:
        input_tensor = input_data.float()

    # Get the spatial dimensions (H, W, D)
    _, _, H, W, D = input_tensor.shape

    # Perform 3D average pooling with a kernel size that covers the entire spatial dimension
    pooled_tensor = F.avg_pool3d(input_tensor, kernel_size=(H, W, D))

    return pooled_tensor


# Example usage
input_data = torch.rand(2, 3, 8, 8, 8)  # (B, C, H, W, D) = (2, 3, 8, 8, 8)
output_data = global_average_pooling_3d(input_data)

print("Input shape:", input_data.shape)
print("Output shape:", output_data.shape)


