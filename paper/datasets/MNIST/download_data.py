import torch
import torchvision
import numpy as np
import os

current_dir = os.path.dirname(__file__)

# Download MNIST dataset
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)

X_train = dataset.data[:50000]
X_test = dataset.data[-10000:]
y_train = np.array(dataset.targets[:50000])
y_test = np.array(dataset.targets[-10000:])

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

#binarize data
X_train = np.array(X_train)/256
X_test = np.array(X_test)/256
X_train = np.array(X_train > 0.5, dtype=int)
X_test = np.array(X_test > 0.5, dtype=int)

np.savetxt(current_dir + '/x_train.csv', X_train, delimiter=",", fmt='%d')
np.savetxt(current_dir + '/x_test.csv', X_test, delimiter=",", fmt='%d')
np.savetxt(current_dir + '/y_train.csv', y_train, delimiter=",", fmt='%d')
np.savetxt(current_dir + '/y_test.csv', y_test, delimiter=",", fmt='%d')




