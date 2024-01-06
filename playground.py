import torch

a = 4.5
t = torch.tensor(a).unsqueeze(0)

print(t)
print(t.dtype)
print(t.shape)
