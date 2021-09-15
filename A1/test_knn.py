import torch
from knn import *
import eecs598
import torchvision
import matplotlib.pyplot as plt
import statistics
import random
from torchvision.utils import make_grid

plt.rcParams['figure.figsize'] = {10.0, 8.0}
plt.rcParams['font.size'] = 16
torch.manual_seed(0)
'''
load data
'''
# x_train, y_train, x_test, y_test = eecs598.data.cifar10()
x_train, y_train, x_test, y_test = eecs598.data.cifar10(500, 250)
# help(eecs598.data.cifar10)

print('Training set:', )
print('  data shape:', x_train.shape)
print('  labels shape: ', y_train.shape)
print('Test set:')
print('  data shape: ', x_test.shape)
print('  labels shape', y_test.shape)

'''
show some images
'''
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# samples_per_class = 12
# samples = []
# for y, cls in enumerate(classes):
#     plt.text(-4, 34 * y + 18, cls, ha='right')
#     idxs, = (y_train == y).nonzero(as_tuple=True)
#     for i in range(samples_per_class):
#         idx = idxs[random.randrange(idxs.shape[0])].item()
#         samples.append(x_train[idx])
# img = torchvision.utils.make_grid(samples, nrow=samples_per_class)
# plt.imshow(eecs598.tensor_to_image(img))
# plt.axis('off')
# plt.show()

'''
implement
'''
dists = compute_distances_two_loops(x_train, x_test)
print('dists has shape: ', dists.shape)


x_train_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)
x_test_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)

dists_one = compute_distances_one_loop(x_train_rand, x_test_rand)
dists_two = compute_distances_two_loops(x_train_rand, x_test_rand)
difference = (dists_one - dists_two).pow(2).sum().sqrt().item()
print('Difference: ', difference)
if difference < 1e-4:
    print('Good! The distance matrices match')
else:
    print('Uh-oh! The distance matrices are different')

x_train_rand = torch.randn(200, 3, 16, 16, dtype=torch.float64)
x_test_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)

dists_two = compute_distances_two_loops(x_train_rand, x_test_rand)
dists_none = compute_distances_no_loops(x_train_rand, x_test_rand)
print(dists_two.shape)
print(dists_none.shape)
difference = (dists_two - dists_none).pow(2).sum().sqrt().item()
print('Difference: ', difference)
if difference < 1e-4:
  print('Good! The distance matrices match')
else:
  print('Uh-oh! The distance matrices are different')