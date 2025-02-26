import torch

# 清理 GPU 1 的缓存
torch.cuda.set_device(1)
torch.cuda.empty_cache()