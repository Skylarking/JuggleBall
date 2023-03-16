import torch
done = torch.tensor([1,0,0,1,1,1,0])
v = torch.tensor([1,2,3,4,5,6,7,8])

dlt_lst = []

for value, done in zip(v,done):
    dlt = value*(1- done)
    dlt_lst.append(dlt)
print(dlt_lst)
