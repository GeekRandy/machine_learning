import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)

print np.matmul(data, data)
print torch.mm(tensor, tensor)
# print torch.dot(tensor.dot(tensor))