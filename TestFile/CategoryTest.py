import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal,Categorical
import math


a = [[1,1,1],[1,3,-torch.inf]]
a = torch.tensor(a, dtype=torch.float)
print(a)
b = torch.softmax(a, dim=-1)
print(b)
dist = Categorical(b)
print(dist.sample())
