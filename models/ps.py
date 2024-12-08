import numpy as np
from numpy.typing import NDArray


class ModelPS:
    '''
        Modelo de Rede Neural: Perceptron Simples (PS)
    '''

    def __init__(self, lr: float, max_epoch: int, C: int):
        # Definindo taxa de aprendizado do modelo
        self.lr = lr

        # Definindo quantidade máxima de épocas
        self.max_epoch = max_epoch

        # Definindo matriz de confusão
        self.matriz_conf = np.zeros((C, C))

    def training(self, X_train: NDArray, Y_train: NDArray):
        '''
            Treinamento do modelo
        '''
        # Quantidade de características e de amostras
        p, N = X_train.shape

        # Adicionando o viés ao modelo
        X_train = np.concatenate(
            (-np.ones((1, N)), X_train)
        )
        print(X_train[:, 0])
        # Definindo matriz de peso inicial
        w = np.random.random_sample((p + 1, 1))-.5

        # Condições iniciais
        error = True
        epoca = 0

        # Período de ajuste dos valores da reta e dos pesos sinápcticos
        while error and epoca < self.max_epoch:
            error = False
            for t in range(N):
                x_t = X_train[:, t].reshape(p+1, 1)
                u_t = (w.T@x_t)[0, 0]
                y_t = 1 if u_t >= 0 else -1
                d_t = Y_train[0, t]
        return w

    def update_model_precision(self, line: int, col: int, y_estim: int, d_t: int):
        '''
            Atualizar a matriz de confusão
        '''
        self.matriz_conf[line][col] += 1

    def show_metrics(self):
        '''
            Retornar acurácia, sensibilidade e especificidade
        '''
        return

    def testing(self, X_test: NDArray, Y_test: NDArray, w_estim: NDArray):
        '''
            Predições do modelo
        '''
        p, N = X_test.shape
        X_test = np.concatenate(
            (-np.ones((1, N)), X_test)
        )
        for t in range(N):
            x_t = X_test[:, t]
            u_t = (w_estim.T@x_t)[0]
            y_t = 1 if u_t >= 0 else - 1
            d_t = Y_test[0, t]
            self.update_model_precision(0, 0, y_t, d_t)
        return self.show_metrics()

    def execute_model(self):
        pass
