import torch
r = torch.ones([128,1]).float()
dw_mask = torch.zeros([128,1]).float()
vs = r*2
vs_ = r*3
gamma = 0.5
a = r + gamma * vs_ * (1 - dw_mask) - vs
b = r + gamma * vs_ * (1 - dw_mask) * - vs
print(a)
print(b)
print(a==b)