from torch.autograd import  Variable
import torch

one = Variable(torch.ones(10,1))
print(one.type)