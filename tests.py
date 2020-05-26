from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
import my_net
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_breast_cancer

def plot_dataset(X, y, net):

    # Bounds of Data
    xx = np.linspace(min(X[:, 0]), max(X[:, 0]), 200)
    yy = np.linspace(min(X[:, 1]), max(X[:, 1]), 200)

    X11 = []
    X22 = []
    pred = []
    for point1 in xx:
        for point2 in yy:
            X11.append(point1)
            X22.append(point2)
            sample = np.array([point1, point2]).reshape(1,-1)
            sample_prediction = net.predict(sample.T)[0][0]
            pred.append(sample_prediction)


    y = y.reshape(1,len(y))[0]
    plt.scatter(X11, X22, c=pred, s=50, linewidth=1)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='white', cmap="RdBu")
    plt.show()

def test_dataset(data):
    X = data[0]
    y = np.expand_dims(data[1], 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )
    xPass = x_train.T
    yPass = y_train.T
    studentNet = studNet.test_train(xPass, yPass)
    results = studentNet.predict(x_test.T)
    #plot_dataset(X, y, studentNet)
    print(accuracy_score(y_test.T[0], results[0]))


# test_dataset(load_breast_cancer(return_X_y=True))
print("Make Moons 0.1:")
test_dataset(datasets.make_moons(n_samples=2500, noise=0.1))
print("Make Moons 0.2:")
test_dataset(datasets.make_moons(n_samples=2500, noise=0.2))
print("Make Moons 0.3:")
test_dataset(datasets.make_moons(n_samples=2500, noise=0.3))
print("Make Moons 0.4:")
test_dataset(datasets.make_moons(n_samples=2500, noise=0.4))
print("Make Moons 0.5:")
test_dataset(datasets.make_moons(n_samples=2500, noise=0.5))
print("Make Moons 0.6:")
test_dataset(datasets.make_moons(n_samples=2500, noise=0.6))
print("Make Moons 0.7:")
test_dataset(datasets.make_moons(n_samples=2500, noise=0.7))
print("Make Moons 0.8:")
test_dataset(datasets.make_moons(n_samples=2500, noise=0.8))
print("Make Circles 0.1:")
test_dataset(datasets.make_circles(n_samples=2000, noise=0.025))
