import numpy as np

def load_data(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        row = line.strip().split(',')
        data.append([float(x) for x in row])

    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


def normalize_labels(y):
    classes = np.unique(y)
    label_map = {c: i for i, c in enumerate(classes)}
    y_new = np.array([label_map[i] for i in y])
    return y_new, len(classes)


def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def one_hot(y, num_classes):
    m = len(y)
    y_encoded = np.zeros((m, num_classes))
    for i in range(m):
        y_encoded[i, y[i]] = 1
    return y_encoded
