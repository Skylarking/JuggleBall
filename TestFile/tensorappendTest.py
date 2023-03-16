import torch

a = torch.ones([64,2])
b = torch.zeros([64,3])

c = torch.concat([a,b],dim=-1)
print(c.shape)