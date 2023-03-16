import torch
a = [0.3573, 0.3603, 0.2824]
a = torch.tensor(a)
b = torch.log(a).sum(dim=-1)
print(b)