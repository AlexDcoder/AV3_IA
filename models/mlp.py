import numpy as np
from numpy.typing import NDArray


class ModelMLP:
    def __init__(
            self, lr: float, max_epoch: int, precis: float, L: int, m: int,
            qs: list[int], C: int):
        # Definindo o valor da taxa de aprendizado
        self.lr = lr

        # Definindo a quantidade máxima de épocas maxEpoch.
        self.max_epoch = max_epoch

        # Definindo o critério de parada em função do erro (EQM)
        self.precis = precis

        # Definindo quantidade de camadas ocultas
        self.L = L

        # Definindo quantidade de neurônios na camada de saída
        self.m = m

        # Definindo número de neurônios por camada oculta
        self.q = np.copy(qs)

        # Definindo matriz de confusão
        self.matriz_conf = np.zeros((C, C))

    @staticmethod
    def foward(x_amostra: NDArray):
        p, N = x_amostra
        for j in range(N):
            if j == 0:
                pass

    @staticmethod
    def backward(x_amostra: NDArray, d):
        pass

    @staticmethod
    def sigmoid(u):
        return (1 - np.exp(-u)) / (1 + np.exp(-u))

    @staticmethod
    def eqm(X: NDArray, Y: NDArray, w: NDArray):
        p_1, N = X.shape
        eq = 0
        for t in range(N):
            x_t = X[:, t].reshape(p_1, 1)
            u_t = w.T@x_t
            d_t = Y[0, t]
            eq += (d_t-u_t[0, 0])**2
        return eq/(2*N)

    def training(self, X_train: NDArray, Y_train: NDArray):
        # Criar uma lista (list) dos elementos: W, u, y, δ cada uma com L + 1 posições.
        grad_local = np.zeros((self.L + 1, 1))
        # Inicializar as L + 1 matrizes W com valores aleatórios pequenos (−0.5, 0.5).
        W = np.random.random_sample((self.L + 1, 1))-.5

        # Adicionar o vetor linha de −1 na primeira linha da matriz de dados Xtreino, resultando em Xtreino ∈ R(p+1)×N
        p, N = X_train.shape
        X_train = np.concatenate(
            (-np.ones((1, N)), X_train)
        )
        epoca = 0
        EQM1 = 1
        EQM2 = 0
        while epoca < self.max_epoch and abs(EQM1 - EQM2) > self.precis:
            # EQM1 = self.eqm(X_train, Y_train, w)
            for t in range(N):
                x_t = X_train[:, t]
                self.foward(x_t)
                d_t = Y_train[0, t]
                self.backward(x_t, d_t)
            epoca += 1
            EQM2 = self.eqm(X_train, Y_train, W)
        return W

    def testing(self, X_test: NDArray, Y_test: NDArray, w_estim: NDArray):
        p, N = X_test.shape
        for t in range(N):
            x_t = X_test[:, t]
            self.foward(x_t)
