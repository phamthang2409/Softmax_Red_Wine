import numpy as np

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def train_softmax(X, y, num_classes, lr=0.01, epochs=1000):
    m, n = X.shape
    W = np.zeros((n, num_classes))
    b = np.zeros((1, num_classes))

    y_onehot = np.zeros((m, num_classes))
    for i in range(m):
        y_onehot[i, y[i]] = 1

    for epoch in range(epochs):
        z = np.dot(X, W) + b
        y_hat = softmax(z)

        dW = (1/m) * np.dot(X.T, (y_hat - y_onehot))
        db = (1/m) * np.sum(y_hat - y_onehot, axis=0, keepdims=True)

        W -= lr * dW
        b -= lr * db

        if epoch % 200 == 0:
            loss = -np.mean(np.sum(y_onehot * np.log(y_hat + 1e-9), axis=1))
            print(f"Epoch {epoch}, Loss = {loss:.4f}")

    return W, b


def predict(X, W, b):
    z = np.dot(X, W) + b
    y_hat = softmax(z)
    return np.argmax(y_hat, axis=1)
