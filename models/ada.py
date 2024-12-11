import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Tuple, List


class ModelADALINE:
    def __init__(self, lr: float, max_epoch: int, precis: float, C: int):
        """
        Inicializa o modelo ADALINE (Adaptive Linear Neuron)

        Args:
            lr (float): Taxa de aprendizado
            max_epoch (int): Número máximo de épocas de treinamento
            precis (float): Precisão para critério de parada
            C (int): Número de categorias/classes
        """
        # Taxa de aprendizado para atualização dos pesos
        self.lr: float = lr

        # Número máximo de iterações de treinamento
        self.max_epoch: int = max_epoch

        # Precisão para determinar convergência
        self.precis: float = precis

        # Número de categorias no problema de classificação
        self.C: int = C

        # Matriz de confusão para avaliar desempenho do modelo
        self.matriz_conf: NDArray = np.zeros((C, C))

    @staticmethod
    def mean_squared_error(X: NDArray, Y: NDArray, w: NDArray) -> float:
        """
        Calcula o Erro Quadrático Médio (MSE)

        Args:
            X (NDArray): Dados de entrada com termo de bias
            Y (NDArray): Dados de destino em codificação one-hot
            w (NDArray): Matriz de pesos

        Returns:
            float: Erro Quadrático Médio
        """
        # Obtém o número de amostras
        _, N = X.shape

        # Inicializa erro total
        total_error: float = 0

        # Computa erro para cada amostra
        for t in range(N):
            x_t = X[:, t]
            y_t = Y[:, t]
            # Calcula vetor de predição
            prediction = w.T @ x_t
            # Computa erro quadrado para todas as classes
            error = np.sum((y_t - prediction)**2)
            total_error += error

        # Calcula MSE
        mse = total_error / (2 * N)

        return mse

    def training(self, X_train: NDArray, Y_train: NDArray) -> NDArray:
        """
        Método de treinamento do ADALINE

        Args:
            X_train (NDArray): Dados de entrada de treinamento
            Y_train (NDArray): Dados de destino de treinamento

        Returns:
            NDArray: Matriz de pesos treinada
        """
        p, N = X_train.shape

        # Adiciona termo de bias
        X_train_with_bias = np.vstack((-np.ones(N), X_train))

        # Inicializa pesos aleatoriamente
        w = np.random.uniform(-0.5, 0.5, (p + 1, self.C))

        # Loop de treinamento
        epoch = 0
        mse_1 = 1
        mse_2 = 0

        # Critério de parada: máximo de épocas ou convergência
        while epoch < self.max_epoch and abs(mse_1 - mse_2) > self.precis:
            # Calcula MSE atual
            mse_1 = self.mean_squared_error(X_train_with_bias, Y_train, w)
            self.learning_hist.append(float(mse_1))

            # Atualização de pesos para cada amostra
            for i in range(N):
                x_t = X_train_with_bias[:, i:i+1]
                u_t = w.T@x_t
                d_t = Y_train[:, i:i+1]
                e_t = d_t - u_t
                w = w + self.lr*(x_t@e_t.T)

            # Calcula novo MSE
            mse_2 = self.mean_squared_error(X_train_with_bias, Y_train, w)
            epoch += 1
        self.learning_hist.append(mse_2)
        return w

    def update_model_precision(self, y_estim: int, d_t: int) -> None:
        """
        Atualiza matriz de confusão

        Args:
            y_estim (int): Classe estimada
            d_t (int): Classe verdadeira
        """
        self.matriz_conf[d_t, y_estim] += 1

    def _show_metrics(self) -> float:
        """
        Calcula métricas de desempenho

        Returns:
            float: Acurácia do modelo
        """
        total = self.matriz_conf.sum()
        true_positives = np.diag(self.matriz_conf)
        accuracy = true_positives.sum() / total

        return accuracy

    def testing(self, X_test: NDArray, Y_test: NDArray, w_estim: NDArray) -> Tuple[NDArray, float, List[float]]:
        """
        Método de teste do modelo

        Args:
            X_test (NDArray): Dados de entrada de teste
            Y_test (NDArray): Dados de destino de teste
            w_estim (NDArray): Pesos estimados

        Returns:
            Tuple contendo:
            - Matriz de confusão
            - Acurácia
            - Histórico de aprendizado
        """
        p, N = X_test.shape

        # Adiciona termo de bias
        X_test_with_bias = np.vstack((-np.ones(N), X_test))

        # Reinicia matriz de confusão
        self.matriz_conf = np.zeros((self.C, self.C))

        # Predição e atualização da matriz de confusão
        for t in range(N):
            x_t = X_test_with_bias[:, t].reshape(p+1, 1)
            u_t = w_estim.T @ x_t

            # Predição multiclasse
            y_t = np.argmax(u_t)
            d_t = np.argmax(Y_test[:, t])

            self.update_model_precision(int(y_t), int(d_t))

        return self.matriz_conf, self._show_metrics(), self.learning_hist
