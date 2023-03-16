import numpy as np

a = np.array([1,2,3,4])
b = np.array([1,2])
c = [a,b]
print(a.shape,b.shape)
d = np.concatenate(c)
print(d.shape)
print(d)