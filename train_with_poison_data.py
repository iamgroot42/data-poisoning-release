import numpy as np
from sklearn.svm import LinearSVC
import os


def load_data():
    dataset_path = os.path.join("../../files/data/mnist_17_train_test.npz")
    f = np.load(dataset_path)

    x_train = f['X_train']
    y_train = f['Y_train'].reshape(-1)
    x_test = f['X_test']
    y_test = f['Y_test'].reshape(-1)

    return (x_train, y_train), (x_test, y_test), (0, 1)


if __name__ == "__main__":
    import sys

    (x_train, y_train), (x_test, y_test), (min_val, max_val) = load_data()

    d = np.load(sys.argv[1])
    x_poison, y_poison = d['X'], d['Y']

    # Fit without poison
    weight_decay = 0.09
    C = 1 / (x_train.shape[0] * weight_decay)
    classifier = LinearSVC(loss='hinge', C=C, tol=1e-10, max_iter=100)
    classifier.fit(x_train, y_train)

    tr_acc = classifier.score(x_train, y_train)
    print("Clean Model | Train accuracy: %.3f" % tr_acc)

    te_acc = classifier.score(x_test, y_test)
    print("Clean Model | Test accuracy: %.3f" % te_acc)

    x_use = np.concatenate((x_train, x_poison), 0)
    y_use = np.concatenate((y_train, y_poison), 0)

    C = 1 / (x_use.shape[0] * weight_decay)
    classifier = LinearSVC(loss='hinge', C=C, tol=1e-10, max_iter=100)
    classifier.fit(x_use, y_use)

    tr_acc = classifier.score(x_use, y_use)
    print("Poisoned Model | Train accuracy: %.3f" % tr_acc)

    te_acc = classifier.score(x_test, y_test)
    print("Poisoned Model | Test accuracy: %.3f" % te_acc)
