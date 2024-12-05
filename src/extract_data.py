import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, file_name: str, C: int, width: int, height: int):
        self.data = FileExplorer(file_name)
        self.width, self.height = width, height
        self.C = C
        self.X = np.empty((width*height, 0))
        self.Y = np.empty((C, 0))
        self.read_and_resize_img()

    def read_and_resize_img(self):
        for i, img_dir in enumerate(self.data.img_dirs.values()):
            for img in (img_dir):
                img_matrix = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                redim_img = cv2.resize(img_matrix, (self.width, self.height))
                x = redim_img.flatten()
                self.X = np.concatenate((
                    self.X,
                    x.reshape(self.width*self.height, 1)
                ), axis=1)
                y = -np.ones((self.C, 1))
                y[i, 0] = 1
                self.Y = np.concatenate((
                    self.Y,
                    y
                ), axis=1)


class Montecarlo:
    def __init__(self, matrix_data: MatrixData, percent_train: float):
        self.matrix_data = matrix_data
        self.percent_train = percent_train

    def execute(
            self, R, ps_hiperparams: list[int | float],
            ada_hiperparams: list[int | float],
            mlp_hiperparams: list[int | float | list[int]]):
        coppied_data = np.concatenate(self.matrix_data.X, self.matrix_data.Y)
        for i in range(R):
            np.random.shuffle(coppied_data)
            X_train = None
            Y_train = None
            X_test = None
            Y_test = None
