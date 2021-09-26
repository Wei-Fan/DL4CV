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


dists = torch.tensor([
    [0.3, 0.4, 0.1],
    [0.1, 0.5, 0.5],
    [0.4, 0.1, 0.2],
    [0.2, 0.2, 0.4],
    [0.5, 0.3, 0.3],
])
y_train = torch.tensor([0, 1, 0, 1, 2])
y_pred_expected = torch.tensor([1, 0, 0])
y_pred = predict_labels(dists, y_train, k=3)
correct = y_pred.tolist() == y_pred_expected.tolist()
print('Correct: ', correct)

num_test = 10000
num_train = 20
num_classes = 5

# Generate random training and test data
torch.manual_seed(128)
x_train = torch.rand(num_train, 3)
y_train = torch.randint(num_classes, size=(num_train,))
x_test = torch.rand(num_test, 3)
classifier = KnnClassifier(x_train, y_train)

# Plot predictions for different values of k
for k in [1, 3, 5]:
    y_test = classifier.predict(x_test, k=k)
    plt.gcf().set_size_inches(8, 8)
    class_colors = ['r', 'g', 'b', 'k', 'y']
    train_colors = [class_colors[c] for c in y_train]
    test_colors = [class_colors[c] for c in y_test]
    plt.scatter(x_test[:, 0], x_test[:, 2],
                color=test_colors, marker='o', s=32, alpha=0.05)
    plt.scatter(x_train[:, 0], x_train[:, 2],
                color=train_colors, marker='*', s=128.0)
    plt.title('Predictions for k = %d' % k, size=16)
    plt.show()

torch.manual_seed(0)
num_train = 5000
num_test = 500
x_train, y_train, x_test, y_test = eecs598.data.cifar10(num_train, num_test)

classifier = KnnClassifier(x_train, y_train)
classifier.check_accuracy(x_test, y_test, k=5)

torch.manual_seed(0)
num_train = 5000
num_test = 500
x_train, y_train, x_test, y_test = eecs598.data.cifar10(num_train, num_test)

k_to_accuracies = knn_cross_validate(x_train, y_train, num_folds=5)

for k, accs in sorted(k_to_accuracies.items()):
  print('k = %d got accuracies: %r' % (k, accs))

ks, means, stds = [], [], []
torch.manual_seed(0)
for k, accs in sorted(k_to_accuracies.items()):
  plt.scatter([k] * len(accs), accs, color='g')
  ks.append(k)
  means.append(statistics.mean(accs))
  stds.append(statistics.stdev(accs))
plt.errorbar(ks, means, yerr=stds)
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.title('Cross-validation on k')
plt.show()

best_k = 1
torch.manual_seed(0)

best_k = knn_get_best_k(k_to_accuracies)
print('Best k is ', best_k)

classifier = KnnClassifier(x_train, y_train)
classifier.check_accuracy(x_test, y_test, k=best_k)