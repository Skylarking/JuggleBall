import torch
import numpy as np


dlt = torch.tensor([1,2,3,4])
done = torch.tensor([True,False,False,True])
# done = done.cpu().flatten().numpy()
print(1-done.float())
print(done)

# for dlt, done in zip(dlt.cpu().flatten().numpy()[::-1], done.cpu().flatten().numpy()[::-1]):
#     print(dlt)
#     print(done)