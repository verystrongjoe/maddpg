import torch
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

x = np.ones(1)
x = torch.from_numpy(x)

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

import torch
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
out.backward()


x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

print(x.grad)



import torch
import numpy as np

x = torch.randn(3, requires_grad=True)
y = x * 2


while y.data.norm() < 1000:
    y = y * 2

print(y)

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

print(x.grad)


import numpy as np
import torch, torch.nn as nn
from torch.autograd import Variable


random_input = Variable(torch.FloatTensor(5, 1, 1).normal_(), requires_grad=False)
print(random_input[:, 0, 0])

# initialize a birectional GRU layer
bi_grus = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=True)

# initialize a GRU layer ( for feeding the sequence reversely)
reverse_gru = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=False)

