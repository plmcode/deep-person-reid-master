import torch
v1 = torch.range(1, 4)
v2 = v1.view(2, 2)
print(v2)
v3 = v2.view(4,-1)
print(v3)