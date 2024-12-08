import os
import cv2
import numpy as np
from models.ps import ModelPS
from models.ada import ModelADALINE
from models.mlp import ModelMLP


class FileExplorer:
    '''
        Nessa classe, foco para extrair quais as pastas e os caminhos
        relativos dos arquivos.
    '''

    def __init__(self, dir_name: str):
        # Caminho da pasta inicial
        self.path = os.path.relpath(dir_name)

        # Nomes das pastas e os caminhos dos seus arquivos
        self.img_dirs = {d: self.copy_imgs(d) for d in os.listdir(self.path)}

    def copy_imgs(self, dir_name: str):
        sub_dir = os.path.join(self.path, dir_name)
        images = []
        with os.scandir(sub_dir) as arquivos:
            for arquivo in arquivos:
                if os.path.isfile(arquivo):
                    images.append(os.path.join(sub_dir, arquivo.name))
        return images


class MatrixData:
    '''
        Nesta classe transformamo as imagem em um conjunto de dados
    '''

    def __init__(self, file_name: str, C: int, dimension: int):
        # Caminhos das imagens
        self.data = FileExplorer(file_name)

        # Definindo dimensões
        self.dimension = dimension

        # Definindo número de rótulos
        self.C = C

        # Variável X será a matriz de dados de dimensões p x N.
        self.X = np.empty((dimension*dimension, 0))

        # Variável Y será a matriz de rótulos
        self.Y = np.empty((C, 0))
        self.read_and_resize_img()

    def read_and_resize_img(self):
        for i, imgs in enumerate(self.data.img_dirs.values()):
            for img in imgs:

                # Transformando a imagem em matriz e redimensionando-a
                img_matrix = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                redim_img = cv2.resize(
                    img_matrix, (self.dimension, self.dimension))

                # Vetorizando imagem
                x = redim_img.flatten()

                # Empilhando amostra para criar a matriz X que terá dimensão p x N
                self.X = np.concatenate((
                    self.X,
                    x.reshape(self.dimension**2, 1)
                ), axis=1)

                # Transformando categorias em vetores
                y = -np.ones((self.C, 1))
                y[i, 0] = 1

                # Empilhando amostra para criar a matriz Y que terá dimensão C x N
                self.Y = np.concatenate((
                    self.Y,
                    y
                ), axis=1)


class Montecarlo:
    def __init__(self, matrix_data: MatrixData, percent_train: float):
        self.matrix_data = matrix_data
        self.percent_train = percent_train

    @staticmethod
    def normalize_data(X: np.ndarray) -> np.ndarray:
        '''
            Normalizar os dados do modelo apresentado
        '''
        return 2 * ((X - np.min(X)) / (np.max(X) - np.min(X))) - 1

    def execute(
            self, R: int,
            ps_hiperp: tuple[float, int, int],
            ada_hiperp: tuple[float, int, float, int],
            mlp_hiperp: tuple[float, int, float, int, int, list[int], int]):

        p, N = self.matrix_data.X.shape

        # Realizar normalização da matriz de entradas
        self.matrix_data.X = Montecarlo.normalize_data(
            self.matrix_data.X)

        # Hiperparâmetros dos modelos de redes neurais
        lr_ps, max_epoch_ps, C = ps_hiperp
        lr_ada, max_epoch_ada, precis_ada, C = ada_hiperp
        lr_mlp, max_epoch_mlp, precis_mlp, L, m, qs, C = mlp_hiperp

        ps = ModelPS(lr_ps, max_epoch_ps, C)
        ada = ModelADALINE(lr_ada, max_epoch_ada, precis_ada, C)
        mlp = ModelMLP(lr_mlp, max_epoch_mlp, precis_mlp, L, m, qs, C)

        # Realizar validação dos modelos em R tempos
        for i in range(R):
            random_index = np.random.permutation(N)
            self.matrix_data.X = self.matrix_data.X[:, random_index]
            self.matrix_data.Y = self.matrix_data.Y[:, random_index]

            X_train = self.matrix_data.X[:, :int(N*self.percent_train)]
            Y_train = self.matrix_data.Y[:, :int(N*self.percent_train)]

            w_ps = ps.training(X_train, Y_train)
            w_ada = ada.training(X_train, Y_train)
            w_mlp = mlp.training(X_train, Y_train)

            X_test = self.matrix_data.X[:, int(N*self.percent_train):]
            Y_test = self.matrix_data.Y[:, int(N*self.percent_train):]

            ps.testing(X_test, Y_test, w_ps)
            ada.testing(X_test, Y_test, w_ada)
            mlp.testing(X_test, Y_test, w_mlp)


if __name__ == '__main__':
    data = MatrixData('RecFac', 20, 10)
    model_1 = ModelPS(.001, 10**3, 20)

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
    model_1.testing(X_test, Y_test, w_estim)
