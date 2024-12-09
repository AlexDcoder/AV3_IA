import numpy as np
from numpy.typing import NDArray
import pandas as pd


class ModelPS:
    def __init__(self, lr: float, max_epoch: int, C: int):
        self.lr = lr
        self.C = C
        self.max_epoch = max_epoch
        self.matriz_conf = np.zeros((C, C))

    def training(self, X_train: NDArray, Y_train: NDArray):
        p, N = X_train.shape

        # Vectorized bias addition
        X_train_with_bias = np.vstack((-np.ones(N), X_train))

        # More efficient weight initialization
        w = np.random.uniform(-0.5, 0.5, (p + 1, self.C))

        # Vectorized training approach
        for _ in range(self.max_epoch):
            # Compute activations for all samples
            u_t = w.T @ X_train_with_bias
            y_t = np.argmax(u_t, axis=0)
            d_t = np.argmax(Y_train, axis=0)

            # Check for error
            errors = d_t != y_t
            if not np.any(errors):
                break

            # Efficient weight update
            for i in range(N):
                if errors[i]:
                    w += (self.lr * (d_t[i] - y_t[i]) *
                          X_train_with_bias[:, i])[:, np.newaxis] / 2

        return w

    def update_model_precision(self, y_estim: int, d_t: int):
        self.matriz_conf[y_estim, d_t] += 1
        if y_estim != d_t:
            self.matriz_conf[d_t, y_estim] += 1

    def show_metrics(self):
        # More efficient metrics computation
        total = self.matriz_conf.sum()
        vp = np.diag(self.matriz_conf)
        fn = self.matriz_conf.sum(axis=1) - vp
        fp = self.matriz_conf.sum(axis=0) - vp
        vn = total - (vp + fn + fp)

        accuracy = vp.sum() / total
        sensitivity = np.mean(vp / (vp + fn))
        specificity = np.mean(vn / (vn + fp))

        return accuracy, sensitivity, specificity

    def testing(self, X_test: NDArray, Y_test: NDArray, w_estim: NDArray):
        p, N = X_test.shape
        X_test_with_bias = np.vstack((-np.ones(N), X_test))

        # Vectorized testing
        u_t = w_estim.T @ X_test_with_bias
        y_t = np.argmax(u_t, axis=0)
        d_t = np.argmax(Y_test, axis=0)

        # Update confusion matrix for all samples
        for y, d in zip(y_t, d_t):
            self.update_model_precision(y, d)

        return self.show_metrics()
