import numpy as np


class ModelMLP:
    def __init__(self, L: int, m: int, qs: list[int], C: int):
        # Definindo quantidade de camadas ocultas
        self.L = L
        # Definindo quantidade de neurônios na camada de saída
        self.m = m

        # Definindo número de neurônios por camada oculta
        self.q = qs

        # Definindo matriz de confusão
        self.matriz_conf = np.zeros((C, C))

    @staticmethod
    def neuron_per_layer(*args):
        return []

    @staticmethod
    def normalize_data(X: np.ndarray):
        '''
            Normalizar os dados do modelo apresentado
        '''
        return 2 * ((X - np.min(X)) / (np.max(X) - np.min(X))) - 1

    @staticmethod
    def training(X_train, Y_train):
        pass

    @staticmethod
    def predict(X_test, Y_test):
        pass
