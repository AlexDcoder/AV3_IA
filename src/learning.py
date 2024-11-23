import numpy as np
import matplotlib.pyplot as plt


class Data:
    def __init__(self, file_name):
        # Extraindo informações do arquivo
        self.file_name = file_name
        self.data = np.loadtxt(file_name, delimiter=',')
        self.data = self.data.T
        self.p, self.N = self.data.shape
        self.X = self.data[:self.p - 1, :]
        self.Y = self.data[self.p - 1, :]
        self.categories = list(set(self.Y))

    def show_graph(self):
        # Gerando gráfico com base nas informações
        plt.title(f"Gráfico gerado à partir dos dados do {self.file_name}.")

        for category in self.categories:
            # Desenhar pontos no gráficos
            plt.scatter(self.X[0, self.Y == category],
                        self.X[1, self.Y == category],  edgecolors='k',
                        label=f'Categoria {int(category)}')

        # Definir limites visuais dos eixos x e y
        plt.xlim(np.min(self.X[0, :]) - .5, np.max(self.X[0, :]) + .5)
        plt.ylim(np.min(self.X[1, :]) - .5, np.max(self.X[1, :]) + .5)

        # Fazer desenho da reta inicial
        plt.plot(
            [np.min(self.X[0, :]), np.max(self.X[0, :])],
            [np.min(self.X[1, :]), np.max(self.X[1, :])],
            color='red'
        )
        plt.xlabel(r'$X_1$')
        plt.ylabel(r'$X_2$')
        plt.legend()
        plt.show()


class ModelsRNA:
    '''
        Classe para implementar funções que representam modelos de redes 
        neurais artificiais.
    '''
    @staticmethod
    def perceptron_simples(X, Y, etha):
        p, N = X.shape
        X = np.concatenate((
            -np.ones((1, N)),
            X)
        )

        w = np.random.random_sample((p, 1)) - .5
        erro = True
        epoca = 0
        while erro:
            erro = False
            for t in range(N):
                x_t = X[:, t].reshape(p+1, 1)
                u_t = (w.T@x_t)[0, 0]
                y_t = 1 if u_t >= 0 else -1
                d_t = Y[0, t]
                e_t = d_t - y_t
                w = w + (etha*e_t*x_t)/2
                erro = True if y_t != d_t else False
            epoca += 1

    @staticmethod
    def adaptive_linear_element():
        pass

    @staticmethod
    def perceptron_multipo():
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


if __name__ == '__main__':
    validate = Validation('spiral.csv')
    data = Data('spiral.csv')
    print(np.min(data.X[0, :]), np.max(data.X[0, :]))
    print(np.min(data.X[1, :]), np.max(data.X[1, :]))
    data.show_graph()
