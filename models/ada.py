import numpy as np
from numpy.typing import NDArray


class ModelADALINE:
    def __init__(self, lr: float, max_epoch: int, precis: float, C: int):
        # Learning rate
        self.lr = lr

        # Maximum number of epochs
        self.max_epoch = max_epoch

        # Precision for stopping criterion
        self.precis = precis

        # Number of categories
        self.C = C

        # Confusion matrix
        self.matriz_conf = np.zeros((C, C))

    @staticmethod
    def mean_squared_error(X: NDArray, Y: NDArray, w: NDArray):
        """
        Calculate Mean Squared Error

        Args:
            X (NDArray): Input data with bias
            Y (NDArray): Target data
            w (NDArray): Weight matrix

        Returns:
            float: Mean Squared Error
        """
        p, N = X.shape
        errors = []

        for t in range(N):
            x_t = X[:, t].reshape(p, 1)
            u_t = w.T @ x_t
            d_t = Y[:, t]
            error = np.mean((d_t - u_t)**2)
            errors.append(error)

        return np.mean(errors)

    def training(self, X_train: NDArray, Y_train: NDArray):
        """
        ADALINE Training Method

        Args:
            X_train (NDArray): Training input data
            Y_train (NDArray): Training target data

        Returns:
            NDArray: Trained weight matrix
        """
        p, N = X_train.shape

        # Add bias term
        X_train_with_bias = np.vstack((-np.ones(N), X_train))

        # Initialize weights randomly
        w = np.random.uniform(-0.5, 0.5, (p + 1, self.C))

        # Training loop
        epoch = 0
        prev_mse = float('inf')

        while epoch < self.max_epoch:
            # Calculate current MSE
            current_mse = self.mean_squared_error(
                X_train_with_bias, Y_train, w)

            # Check stopping criterion
            if abs(prev_mse - current_mse) <= self.precis:
                break

            for t in range(N):
                x_t = X_train_with_bias[:, t].reshape(
                    p+1, 1)  # Shape: (p+1, 1)
                u_t = w.T @ x_t  # Shape: (C, 1)
                d_t = Y_train[:, t]  # Shape: (C, 1)

                # Delta rule weight update
                error = d_t - u_t  # Shape: (C, 1)
                # Shape of x_t.T is (1, p+1), so this works for broadcasting
                w += self.lr * error * x_t.T

            prev_mse = current_mse
            epoch += 1

        return w

    def update_model_precision(self, y_estim: int, d_t: int):
        """
        Update confusion matrix

        Args:
            y_estim (int): Estimated class
            d_t (int): True class
        """
        self.matriz_conf[y_estim, d_t] += 1
        if y_estim != d_t:
            self.matriz_conf[d_t, y_estim] += 1

    def show_metrics(self):
        """
        Calculate performance metrics

        Returns:
            tuple: Accuracy, Sensitivity, Specificity
        """
        total = self.matriz_conf.sum()
        true_positives = np.diag(self.matriz_conf)
        false_negatives = self.matriz_conf.sum(axis=1) - true_positives
        false_positives = self.matriz_conf.sum(axis=0) - true_positives
        true_negatives = total - \
            (true_positives + false_negatives + false_positives)

        accuracy = true_positives.sum() / total
        sensitivity = np.mean(
            true_positives / (true_positives + false_negatives))
        specificity = np.mean(
            true_negatives / (true_negatives + false_positives))

        return accuracy, sensitivity, specificity

    def testing(self, X_test: NDArray, Y_test: NDArray, w_estim: NDArray):
        """
        Model Testing Method

        Args:
            X_test (NDArray): Test input data
            Y_test (NDArray): Test target data
            w_estim (NDArray): Estimated weights

        Returns:
            tuple: Performance metrics
        """
        p, N = X_test.shape

        # Add bias term
        X_test_with_bias = np.vstack((-np.ones(N), X_test))

        # Reset confusion matrix
        self.matriz_conf = np.zeros((self.C, self.C))

        # Predict and update confusion matrix
        for t in range(N):
            x_t = X_test_with_bias[:, t].reshape(p+1, 1)
            u_t = w_estim.T @ x_t

            # Multiclass prediction
            y_t = np.argmax(u_t)
            d_t = np.argmax(Y_test[:, t])

            self.update_model_precision(y_t, d_t)

        return self.show_metrics()
