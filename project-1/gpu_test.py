import torch
print(torch.cuda.is_available())        # True가 나와야 함
print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX 2060이 출력되어야 함
