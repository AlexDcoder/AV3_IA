import numpy as np
import matplotlib.pyplot as plt


class Data:
    def __init__(self, file_name):
        # Extraindo informações do arquivo
        self.file_name = file_name
        self.data = np.loadtxt(file_name, delimiter=',').T
        self.p, self.N = self.data.shape
        self.X = self.data[:self.p - 1, :]
        self.Y = self.data[self.p - 1, :]
        self.categories = list(set(self.Y))


class ModelsRNA:
    '''
        Classe para implementar funções que representam modelos de redes
        neurais artificiais.
    '''

    @staticmethod
    def perceptron_simples(X, Y, etha, max_epocas):
        p, N = X.shape
        X = np.concatenate(
            (-np.ones((1, N)), X)
        )

        w = np.random.random_sample((p + 1, 1))-.5
        x_axis = np.linspace(np.min(X[1, :]), np.max(X[1, :]))
        x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
        x2 = np.nan_to_num(x2)

        error = True
        epoca = 0

        while error and epoca < max_epocas:
            error = False
            for t in range(N):
                x_t = X[:, t].reshape(p+1, 1)
                u_t = (w.T@x_t)[0, 0]
                y_t = 1 if u_t >= 0 else -1
                d_t = Y[t]
                e_t = d_t - y_t
                w = w + (etha*e_t*x_t) / 2
                if (y_t != d_t):
                    error = True
            x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
            x2 = np.nan_to_num(x2)
            epoca += 1

        x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
        x2 = np.nan_to_num(x2)

        return x_axis, x2, w

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

    @staticmethod
    def adaptive_linear_element(X, Y, etha, max_epocas, precisao):
        p, N = X.shape
        X = np.concatenate(
            (-np.ones((1, N)), X)
        )

        w = np.random.random_sample((p + 1, 1))-.5
        x_axis = np.linspace(np.min(X[1, :]), np.max(X[1, :]))
        x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
        x2 = np.nan_to_num(x2)

        epoca = 0
        EQM1 = 0
        EQM2 = 0

        while epoca < max_epocas and abs(EQM1 - EQM2) > precisao:
            EQM1 = ModelsRNA.eqm(X, Y, w)
            for t in range(N):
                x_t = X[:, t:t+1]
                u_t = (w.T@x_t)[0, 0]
                d_t = Y[0, t]
                e_t = d_t - u_t
                w = w + etha*e_t*x_t
            epoca += 1
            EQM2 = ModelsRNA.eqm(X, Y, w)
            x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
            x2 = np.nan_to_num(x2)

        x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
        x2 = np.nan_to_num(x2)
        return x_axis, x2, w

    @staticmethod
    def foward(X):
        pass

    @staticmethod
    def backward(X, d):
        pass

    @staticmethod
    def perceptron_multipo(
            X, Y, etha, max_epocas, qtd_cam, qtd_escolh, qtd_saida, precisao):
        p, N = X.shape
        X = np.concatenate(
            (-np.ones((1, N)), X)
        )
        W = np.random.random_sample((p + 1, qtd_cam + 1))-.5
        print(W)
        pass


class Validation:
    '''
        Classe para a validação dos modelos
    '''

    def __init__(self, file_name):
        self.data = Data(file_name)

    def montecarlo(self, percent_train, etha, R):
        coppied_data = np.copy(self.data.data)
        for _ in range(R):
            np.random.shuffle(coppied_data.T)
