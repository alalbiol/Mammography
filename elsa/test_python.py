import torch
print("PyTorch version:", torch.__version__)
print("CUDA version (compiled):", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0))
print("Device compute capability:", torch.cuda.get_device_capability(0))


import torch
x = torch.randn(10000, 10000, device="cuda")
print(x.sum())