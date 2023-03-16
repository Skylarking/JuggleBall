import torch
import torch.nn as nn

class cicr_conv(nn.Module):
	def __init__(self, feature=1, num_kernel=9, kernel_size=3, stride=1, padding=0):
		super(cicr_conv, self).__init__()
		self.net = nn.Conv1d(in_channels=feature, out_channels=num_kernel, kernel_size=kernel_size, stride=stride, padding=padding)
		self.pool = nn.MaxPool1d(kernel_size=2)
	def forward(self, input):
		i = input.permute(0,2,1)	#(b,f,n)
		out = self.net(i)
		out = torch.tanh(out)
		out = self.pool(out)
		out = out.permute(0,2,1)	#(b,n,f)
		return out

s = torch.ones([1,738,1])

net = cicr_conv(padding=1)
out = net(s)

print(out.shape)