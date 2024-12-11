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

        # Definindo quantidade de categorias
        self.C = C

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
    def tanh(u):
        return np.tanh(u)

    @staticmethod
    def tanh_derivative(output):
        return 1 - output ** 2

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

    def sign(self, u_t):
        return u_t

    def training(self, X_train: NDArray, Y_train: NDArray):
        # Criar uma lista (list) dos elementos: W, u, y, δ cada uma com L + 1 posições.
        grad_local = np.zeros((self.L + 1, 1))
        # Inicializar as L + 1 matrizes W com valores aleatórios pequenos (−0.5, 0.5).
        W = np.random.random_sample((self.L + 1, self.C))-.5

        # Adicionar o vetor linha de −1 na primeira linha da matriz de dados Xtreino, resultando em Xtreino ∈ R(p+1)×N
        p, N = X_train.shape
        X_train_with_bias = np.vstack((-np.ones(N), X_train))

        for epoca in range(self.max_epoch):
            current_mse = self.mean_squared_error(
                X_train_with_bias, Y_train, w)
            self.foward(X_train_with_bias)
            d_t = None
            self.backward(X_train, d_t)
            prev_mse = current_mse

        return W

    def testing(self, X_test: NDArray, Y_test: NDArray, w_estim: NDArray):
        p, N = X_test.shape
        for t in range(N):
            x_t = X_test[:, t]
            self.foward(x_t)

    def show_graphs(self):
        pass
