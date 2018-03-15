import matplotlib.pyplot as plt
import numpy as np
import random as ran
# Applying SOM into Mnist data
from tensorflow.examples.tutorials.mnist import input_data
# Import som class and train into 30 * 30 sized of SOM lattice
import os
from model.som import Som

# download dataset mnist
mnist = input_data.read_data_sets("data", one_hot=True)

# actual directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# directory where save the model
directory = "model_mnist"
directory = os.path.join(dir_path, directory)
# dim grid and num epoch
n = 60
m = 60
epoch = 300
result = None

# choose if do test without train
if not os.path.exists(directory):
    os.makedirs(directory)
    test = True
    train = True
else:
    test = True
    train = False


def train_size(num):
    _x_train = mnist.train.images[:num, :]
    _y_train = mnist.train.labels[:num, :]
    return _x_train, _y_train


# get dataset to train, validate and test the model
x_train, y_train = train_size(100)
x_val, y_val = train_size(110)
x_test, y_test = train_size(195)
x_val = x_val[100:110, :]
y_val = y_val[100:110, :]
x_test = x_test[110:195, :]
y_test = y_test[110:195, :]


def display_digit(num):
    label = y_test[num].argmax(axis=0)
    image = x_test[num].reshape([28, 28])
    # plt.title('Example: %d  Label: %d' % (num, label))
    # plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    # plt.show()
    return x_test[num], y_test[num], image, label


# show image choose to test the model
x, lab, image1, label1 = display_digit(ran.randint(0, (x_test.shape[0]-1)))

# init Som class
som = Som(n, m, x_train.shape[1], directory, epoch)

# do train
if train:
    som.train(x_train)

# do test
if test:
    result = som.test_model(directory, x)

# Fit train data into SOM lattice
mapped = som.map_vects(x_train)
mappedarr = np.array(mapped)

x1 = mappedarr[:, 0]
y1 = mappedarr[:, 1]

index = [np.where(r == 1)[0][0] for r in y_train]
index = list(map(str, index))


# Plots: 1)
plt.figure(1, figsize=(12, 6))
plt.subplot(221)
# Plot 1 for Training only
plt.scatter(x1, y1)
# Just adding text
for i, m in enumerate(mapped):
    plt.text(m[0], m[1], index[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.title('Train MNIST 100')

# Valuation
if train:
    mappedtest = som.map_vects(x_val)
    mappedtestarr = np.array(mappedtest)
    x2 = mappedtestarr[:, 0]
    y2 = mappedtestarr[:, 1]

    index2 = [np.where(r == 1)[0][0] for r in y_val]
    index2 = list(map(str, index2))

    plt.subplot(222)
    # Plot 2:
    plt.scatter(x1, y1)
    # Just adding text
    for i, m in enumerate(mapped):
        plt.text(m[0], m[1], index[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.scatter(x2, y2)
    # Just adding text
    for i, m in enumerate(mappedtest):
        plt.text(m[0], m[1], index2[i], ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5, lw=0))

# do test plot
if test:
    mappedtestarr = np.array(result)
    x3 = mappedtestarr[:, 0]
    y3 = mappedtestarr[:, 1]

    index3 = [np.where(r == 1)[0][0] for r in [lab]]
    index3 = list(map(str, index3))

    plt.subplot(223)
    plt.title('Example: Label: %d' % label1)
    plt.imshow(image1, cmap=plt.get_cmap('gray_r'))

    plt.subplot(224)
    # Plot 3: Training + Testing
    plt.scatter(x1, y1)
    # Just adding text
    for i, m in enumerate(mapped):
        plt.text(m[0], m[1], index[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.scatter(x3, y3)
    # Just adding text
    for i, m in enumerate(result):
        plt.text(m[0], m[1], index3[i], ha='center', va='center', bbox=dict(facecolor='blue', alpha=0.4, lw=0))

plt.title('Test MNIST 10 + Train MNIST 100')
plt.show()
