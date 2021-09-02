import torch

def mutate_tensor(x, indices, values):
  for i in range(len(values)):
    x[indices[i]] = values[i]
  return x

x = torch.zeros(3,2)
x[0,1] = 10
x[1,0] = 100
print(x)

indices = [(0,0),(1,0),(0,1)]
values = [3,3,3]
x = mutate_tensor(x,indices,values)
print(x)
