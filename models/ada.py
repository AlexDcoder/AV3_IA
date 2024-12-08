import numpy as np

from numpy.typing import NDArray


class ModelADALINE:
    '''
        Modelo de Rede Neural: Adaptive Linear Element (ADALINE)
    '''

    def __init__(self, lr: float, max_epoch: int, precis: float, C: int):
        # Definindo taxa de aprendizado do modelo
        self.lr = lr

        # Definindo quantidade de máxima de épocas
        self.max_epoch = max_epoch

        # Definindo a precisão do modelo
        self.precis = precis

        # Definindo matriz de confusão
        self.matriz_conf = np.zeros((C, C))

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
        p, N = X_train.shape

        X_train = np.concatenate(
            (-np.ones((1, N)), X_train)
        )

        w = np.random.random_sample((p + 1, 1))-.5

        epoca = 0
        EQM1 = 1
        EQM2 = 0

        while epoca < self.max_epoch and abs(EQM1 - EQM2) > self.precis:
            EQM1 = self.eqm(X_train, Y_train, w)
            for t in range(N):
                x_t = X_train[:, t:t+1]
                u_t = (w.T@x_t)[0, 0]
                d_t = Y_train[0, t]
                e_t = d_t - u_t
                w = w + self.lr*e_t*x_t
            epoca += 1
            EQM2 = self.eqm(X_train, Y_train, w)

        return w

    def update_model_precision(self, y_estim: int, d_t: int):
        '''
            Atualizar a matriz de confusão
        '''
        pass

    def show_metrics(self):
        pass

    def testing(self, X_test: NDArray, Y_test: NDArray, w_estim: NDArray):
        p, N = X_test.shape

        for t in range(N):
            x_t = X_test[:, t]
            u_t = (w_estim.T@x_t)[0, 0]
            y_t = 1 if u_t >= 0 else -1
            print(Y_test[0, t])
            d_t = Y_test[0, t]
            self.update_model_precision(y_t, d_t)
