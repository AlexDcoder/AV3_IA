import numpy as np


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

    @staticmethod
    def normalize_data(X: np.ndarray):
        '''
            Normalizar os dados do modelo apresentado
        '''
        return 2 * ((X - np.min(X)) / (np.max(X) - np.min(X))) - 1

    def update_model_precision(self, y_estim, d_t):
        '''
            Atualizar a matriz de confusão
        '''
        if y_estim == d_t:
            if y_estim == 1:
                self.matriz_conf[0][0] += 1
            else:
                self.matriz_conf[1][1] += 1
        if y_estim < d_t:
            self.matriz_conf[0][1] += 1
        if y_estim > d_t:
            self.matriz_conf[1][0] += 1

    def show_metrics(self):
        acur = self.matriz_conf[0, :] / \
            self.matriz_conf[0, :] + self.matriz_conf[1, :]

        sens = self.matriz_conf[0, 0] / \
            self.matriz_conf[0, 0] + self.matriz_conf[1, 0]

        espec = self.matriz_conf[1, 1] / \
            self.matriz_conf[1, 1] + self.matriz_conf[0, 1]

        return acur, sens, espec

    def training(self, X_train: np.ndarray, Y_train: np.ndarray):
        '''
            Treinamento do modelo
        '''
        # Qunatidade de características e de amostras
        p, N = X_train.shape

        # Adicionando o viés ao modelo
        X_train = np.concatenate(
            (-np.ones((1, N)), X_train)
        )

        # Definindo matriz de peso inicial
        w = np.random.random_sample((p + 1, 1))-.5

        # Reta que representa o modelo do perceptron simples em seu início
        x_axis = np.linspace(np.min(X_train[1, :]), np.max(X_train[1, :]))
        x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
        x2 = np.nan_to_num(x2)

        # Condições iniciais
        error = True
        epoca = 0

        # Periíodo de ajuste dos valores da reta e dos pesos sinápcticos
        while error and epoca < self.max_epoch:
            error = False
            for t in range(N):
                x_t = X_train[:, t].reshape(p+1, 1)
                u_t = (w.T@x_t)[0, 0]
                y_t = 1 if u_t >= 0 else -1
                d_t = Y_train[0, t]
                e_t = d_t - y_t
                w = w + (self.lr*e_t*x_t) / 2
                error = True if y_t != d_t else False

            x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
            x2 = np.nan_to_num(x2)
            epoca += 1

        # Fim do treinamento
        x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
        x2 = np.nan_to_num(x2)

        return w

    @staticmethod
    def predict(X_test: np.ndarray, Y_test: np.ndarray, w_estim: np.ndarray):
        '''
            Predições do modelo
        '''
        p, N = X_test.shape

        for t in range(N):
            x_t = X_test[:, t]
            u_t = (w_estim.T@x_t)[0, 0]
            d_t = Y_test[t]
            print(u_t)
