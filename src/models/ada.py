import numpy as np


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
    def normalize_data(X: np.ndarray):
        '''
            Normalizar os dados do modelo apresentado
        '''
        return 2 * ((X - np.min(X)) / (np.max(X) - np.min(X))) - 1

    @staticmethod
    def eqm(X, Y, w):
        p_1, N = X.shape
        eq = 0
        for t in range(N):
            x_t = X[:, t].reshape(p_1, 1)
            u_t = w.T@x_t
            d_t = Y[0, t]
            eq += (d_t-u_t[0, 0])**2
        return eq/(2*N)

    def training(self, X_train, Y_train):
        p, N = X_train.shape

        X_train = np.concatenate(
            (-np.ones((1, N)), X_train)
        )

        w = np.random.random_sample((p + 1, 1))-.5
        x_axis = np.linspace(np.min(X_train[1, :]), np.max(X_train[1, :]))
        x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
        x2 = np.nan_to_num(x2)

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
            x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
            x2 = np.nan_to_num(x2)

        x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
        x2 = np.nan_to_num(x2)
        return w

    def update_model_precision(self, y_estim, d_t):
        '''
            Atualizar a matriz de confusão
        '''
        pass

    def show_metrics(self):
        acur = self.matriz_conf[0, :] / \
            self.matriz_conf[0, :] + self.matriz_conf[1, :]

        sens = self.matriz_conf[0, 0] / \
            self.matriz_conf[0, 0] + self.matriz_conf[1, 0]

        espec = self.matriz_conf[1, 1] / \
            self.matriz_conf[1, 1] + self.matriz_conf[0, 1]

        return acur, sens, espec

    @staticmethod
    def predict(X_test, Y_test, w_estim):
        p, N = X_test.shape

        for t in range(N):
            x_t = X_test[:, t]
            u_t = (w_estim.T@x_t)[0, 0]
            print(u_t)
