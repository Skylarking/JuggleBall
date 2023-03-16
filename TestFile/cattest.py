import torch
s = [torch.rand([128, 12])]
all_s = torch.concat(s,dim=-1)
print(s[0] == all_s)