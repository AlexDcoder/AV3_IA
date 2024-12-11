import numpy as np


class Layer():
    def _init_(self, neuron, p):
        self.w: np.ndarray = np.random.random_sample((neuron, p+1))-.5
        self.u = None
        self.y = -np.ones((neuron+1, 1))
        self.delta = None

    def getU(self):
        return self.u

    def getY(self):
        return self.y[:-1]

    def getYB(self):
        return self.y

    def getWB(self):
        return self.w[:, :-1].T

    def setU(self, x_amostra: np.ndarray):
        self.u = self.w @ x_amostra

    def setY(self, actY: np.ndarray):
        self.y[:-1] = actY


class NetWork():
    def _init_(self, p, innerLayers, m, learningRate: float = 1e-3):
        layers = [p] + innerLayers + [m]
        self.lyrs = [Layer(layers[i+1], layers[i])
                     for i in range(len(layers)-1)]
        self.lr = learningRate
        self.history = {'loss': []}

    def activation(self, u: np.ndarray) -> np.ndarray:
        return np.tanh(u)

    def derivada_activation(self, u: np.ndarray) -> np.ndarray:
        return 1 - u ** 2

    def foward(self, x_amostra: np.ndarray):
        for i, layer in enumerate(self.lyrs):
            entrada = x_amostra if i == 0 else self.lyrs[i - 1].getYB()
            layer.setU(entrada)
            layer.setY(self.activation(layer.getU()))
        return self.lyrs[-1].getY()

    def backward(self, x_amostra: np.ndarray, d: np.ndarray) -> None:
        for j in range(len(self.lyrs)-1, -1, -1):
            if j == len(self.lyrs) - 1:
                erro = (d - self.lyrs[j].getY())
                self.lyrs[j].delta = self.derivada_activation(
                    self.lyrs[j].getY()) * erro
                self.lyrs[j].w += self.lr * \
                    (self.lyrs[j].delta @ self.lyrs[j-1].getYB().T)
            elif j == 0:
                wb = self.lyrs[j+1].getWB()
                self.lyrs[j].delta = self.derivada_activation(
                    self.lyrs[j].getY()) * (wb @ self.lyrs[j+1].delta)
                self.lyrs[j].w += self.lr * (self.lyrs[j].delta @ x_amostra.T)
            else:
                wb = self.lyrs[j+1].getWB()
                self.lyrs[j].delta = self.derivada_activation(
                    self.lyrs[j].getY()) * (wb @ self.lyrs[j+1].delta)
                self.lyrs[j].w += self.lr * \
                    (self.lyrs[j].delta @ self.lyrs[j-1].getYB().T)

    def EQM(self, x_matriz: np.ndarray, y_matrix: np.ndarray):
        eqm = 0
        for n in range(x_matriz.shape[1]):
            amostra = x_matriz[:, [n]]
            d = y_matrix[:, [n]]
            output = self.foward(amostra)
            eqm += np.sum((d - output)**2)
        eqm /= (2*x_matriz.shape[1])
        return eqm

    def train(self, x_matriz: np.ndarray, y_array: np.ndarray, epoch: int = 5000, min_erro: float = 1e-8):
        last_eqm = np.inf
        for e in range(epoch):
            eqm = 0
            for n in range(x_matriz.shape[1]):
                amostra = x_matriz[:, [n]]
                d = y_array[:, [n]]
                self.foward(amostra)
                self.backward(amostra, d)

            eqm = self.EQM(x_matriz, y_array)
            print(f"Epoch: {e}; EQM: {eqm}")
            self.history['loss'].append(eqm)

            if np.abs(eqm - last_eqm) <= min_erro:
                break
            else:
                last_eqm = eqm

    def sigmoide(self, x: np.ndarray) -> np.ndarray:
        """
        Função sigmoidal para converter qualquer valor em 1 ou -1.
        :param x: Valor de entrada.
        :return: Valor convertido em 1 ou -1.
        """
        return np.where(1 / (1 + np.exp(-x)) > 0.5, 1, -1)

    def predic(self, x_matrix: np.ndarray):
        predic = [np.argmax(self.foward(x_matrix[:, [n]]))
                  for n in range(x_matrix.shape[1])]
        return np.array(predic)
