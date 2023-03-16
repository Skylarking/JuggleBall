import torch

a = torch.tensor([[0,0]]).view(-1)
print(a)
a = torch.tensor([2,3,4]).view(-1,1)
print(a)