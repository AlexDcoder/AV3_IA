# Optimized extract_data.py
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from models.ada import ModelADALINE
from models.ps import ModelPS


class FileExplorer:
    def __init__(self, dir_name: str):
        self.path = os.path.abspath(dir_name)
        self.img_dirs = self.get_image_paths()

    def get_image_paths(self):
        # Use listdir comprehension and faster file checking
        return {
            d: [
                os.path.join(self.path, d, f)
                for f in os.listdir(os.path.join(self.path, d))
                if os.path.isfile(os.path.join(self.path, d, f))
            ]
            for d in os.listdir(self.path)
        }


class MatrixData:
    def __init__(self, file_name: str, C: int, dimension: int):
        self.data = FileExplorer(file_name)
        self.dimension = dimension
        self.C = C
        self.X = None
        self.Y = None
        self.read_and_resize_img()

    def process_image(self, img, category_index):
        # Process single image
        img_matrix = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        redim_img = cv2.resize(img_matrix, (self.dimension, self.dimension))
        x = redim_img.flatten()

        y = -np.ones((self.C, 1))
        y[category_index, 0] = 1

        return x.reshape(self.dimension**2, 1), y

    def read_and_resize_img(self):
        # Use ThreadPoolExecutor for parallel image processing
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, imgs in enumerate(self.data.img_dirs.values()):
                for img in imgs:
                    futures.append(executor.submit(self.process_image, img, i))

            results_x, results_y = zip(
                *[future.result() for future in as_completed(futures)])

        self.X = np.concatenate(results_x, axis=1)
        self.Y = np.concatenate(results_y, axis=1)


class Montecarlo:
    def __init__(self, matrix_data: MatrixData, percent_train: float):
        self.matrix_data = matrix_data
        self.percent_train = percent_train

    @staticmethod
    def normalize_data(X: np.ndarray) -> np.ndarray:
        # More efficient normalization
        return 2 * (X - X.min()) / (X.max() - X.min()) - 1

    def execute(self, R: int, *hyperparameters):
        from models.ps import ModelPS
        from models.ada import ModelADALINE
        from models.mlp import ModelMLP

        p, N = self.matrix_data.X.shape

        # Single normalization call
        self.matrix_data.X = self.normalize_data(self.matrix_data.X)

        # Unpack hyperparameters more efficiently
        lr_ps, max_epoch_ps, lr_ada, max_epoch_ada, precis_ada, \
            lr_mlp, max_epoch_mlp, precis_mlp, L, m, qs = hyperparameters[:12]
        C = self.matrix_data.C

        # Create models once
        ps = ModelPS(lr_ps, max_epoch_ps, C)
        ada = ModelADALINE(lr_ada, max_epoch_ada, precis_ada, C)
        mlp = ModelMLP(lr_mlp, max_epoch_mlp, precis_mlp, L, m, qs, C)

        # Preallocate metrics list
        metrics_ps, metrics_ada, metrics_mlp = [], [], []

        # Vectorized data splitting and randomization
        for _ in range(R):
            random_index = np.random.permutation(N)
            X = self.matrix_data.X[:, random_index]
            Y = self.matrix_data.Y[:, random_index]

            split_point = int(N * self.percent_train)
            X_train, X_test = X[:, :split_point], X[:, split_point:]
            Y_train, Y_test = Y[:, :split_point], Y[:, split_point:]

            # Training and testing
            w_ps = ps.training(X_train, Y_train)
            w_ada = ada.training(X_train, Y_train)
            w_mlp = mlp.training(X_train, Y_train)

            metrics_ps.append(ps.testing(X_test, Y_test, w_ps))
            metrics_ada.append(ada.testing(X_test, Y_test, w_ada))
            metrics_mlp.append(mlp.testing(X_test, Y_test, w_mlp))

        return metrics_ps, metrics_ada, metrics_mlp


if __name__ == '__main__':
    data = MatrixData('RecFac', 20, 50)
    model_1 = ModelADALINE(.01, 10**3, .01, 20)

    p, N = data.X.shape
    random_index = np.random.permutation(N)
    X = data.X[:, random_index]
    Y = data.Y[:, random_index]

    X = Montecarlo.normalize_data(X)
    X_train = X[:, :int(N*.8)]
    Y_train = Y[:, :int(N*.8)]

    X_test = X[:, int(N*.8):]
    Y_test = Y[:, int(N*.8):]
    w_estim = model_1.training(X_train, Y_train)
    print(w_estim)
    model_1.testing(X_test, Y_test, w_estim)
