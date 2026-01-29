import numpy as np
from utils import load_data, normalize_labels, standardize
from model import train_softmax, predict
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main():
    X, y = load_data("data/winequality-red.csv")
    y, num_classes = normalize_labels(y)
    X = standardize(X)

    W, b = train_softmax(X, y, num_classes, lr=0.05, epochs=2000)

    y_pred = predict(X, W, b)
    print("Training accuracy:", accuracy(y, y_pred))


if __name__ == "__main__":
    main()
