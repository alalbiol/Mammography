import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
a = torch.rand(10,10,device='cuda')
print(a)
print(a.device)
print(a.shape)
print(a.dtype)
print(a.requires_grad)
print(a**2)
