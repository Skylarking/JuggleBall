import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal,Categorical
import math

'''
循环卷积
	输入：(B,N,F),N是多少个feature，F是feature维数，F=1
'''
class cicr_conv(nn.Module):
    def __init__(self, feature=1, num_kernel=9, kernel_size=3, stride=1, padding=0):
        super(cicr_conv, self).__init__()
        self.net = nn.Conv1d(in_channels=feature, out_channels=num_kernel, kernel_size=kernel_size, stride=stride,
                             padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, input):
        i = input.permute(0, 2, 1)  # 变为(b,f,n)
        out = self.net(i)
        out = torch.tanh(out)
        out = self.pool(out)
        out = out.permute(0, 2, 1)  # 变为(b,n,f)
        return out


''' 
实体concate，确保他们feature长度一致
    输入: in1(B,N1,F)	 in1(B,N2,F)，F要一致
    输出: out(B,N1+N2,F)
'''
class EntityConcat(nn.Module):
    def __init__(self):
        super(EntityConcat, self).__init__()

    def forward(self, inp1, inp2):
        out = torch.concat([inp1,inp2], dim=1)
        return out


'''
编码层，将实体编码到netwidth维度
    输入: 

'''
class Dense(nn.Module):
    def __init__(self, inp_dim, netwidth=256):
        super(Dense, self).__init__()
        self.C1 = nn.Linear()

    def forward(self, inp):
        pass
