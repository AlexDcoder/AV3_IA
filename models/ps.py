import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple


class ModelPS:
    def __init__(self, lr: float, max_epoch: int, C: int):
        """
        Inicializa o modelo Perceptron

        Args:
            lr (float): Taxa de aprendizado
            max_epoch (int): Número máximo de épocas de treinamento
            C (int): Número de rótulos/classes
        """
        # Taxa de aprendizado para atualização dos pesos
        self.lr: float = lr

        # Número de classes no problema de classificação
        self.C: int = C

        # Número máximo de iterações de treinamento
        self.max_epoch: int = max_epoch

        # Matriz de confusão para avaliar desempenho do modelo
        self.matriz_conf: NDArray = np.zeros((C, C))

        # Histórico de erros de aprendizado por época
        self.learning_hist: List[NDArray] = []

    @staticmethod
    def _sign(x: float) -> int:
        """
        Função de ativação de sinal (degrau bipolar)

        Args:
            x (float): Valor de entrada

        Returns:
            int: 1 se x >= 0, -1 caso contrário
        """
        return 1 if x >= 0 else -1

    def training(self, X_train: NDArray, Y_train: NDArray) -> NDArray:
        """
        Método de treinamento do Perceptron

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

        # Variáveis de controle de treinamento
        epoch = 0
        erro = True

        # Treinamento iterativo
        while erro and epoch < self.max_epoch:
            erro = False
            epoch_errors = 0.0

            # Itera sobre todas as amostras
            for t in range(N):
                x_t = X_train_with_bias[:, t:t+1]
                u_t = w.T@x_t

                # Aplica função de ativação de sinal
                y_t = np.array(
                    list(map(ModelPS._sign, u_t))
                ).reshape(self.C, 1)

                # Obtém vetor de destino
                d_t = Y_train[:, t:t+1]

                # Calcula erro
                e_t = d_t - y_t

                # Atualiza pesos usando regra de aprendizado do Perceptron
                w = w + self.lr*(x_t@e_t.T) / 2

                # Verifica se houve erro na classificação
                erro = True if np.any(y_t != d_t) else False

                # Acumula erro quadrático
                epoch_errors += e_t**2

            # Armazena histórico de erros da época
            self.learning_hist.append(epoch_errors)

            # Incrementa contador de épocas
            epoch += 1

        return w

    def _update_model_precision(self, y_estim: int, d_t: int) -> None:
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
        # Calcula total de amostras
        total = self.matriz_conf.sum()

        # Obtém valores de verdadeiros positivos na diagonal
        vp = np.diag(self.matriz_conf)

        # Calcula acurácia
        accuracy = vp.sum() / total
        return accuracy

    def testing(self, X_test: NDArray, Y_test: NDArray,
                w_estim: NDArray) -> Tuple[NDArray, float, List[NDArray]]:
        """
        Método de teste do modelo Perceptron

        Args:
            X_test (NDArray): Dados de entrada de teste
            Y_test (NDArray): Dados de destino de teste
            w_estim (NDArray): Pesos estimados no treinamento

        Returns:
            Tuple contendo:
            - Matriz de confusão
            - Acurácia do modelo
            - Histórico de erros de aprendizado
        """
        # Obtém número de amostras de teste
        _, N = X_test.shape

        # Adiciona termo de bias
        X_test_with_bias = np.vstack((-np.ones(N), X_test))

        # Reinicia matriz de confusão
        self.matriz_conf = np.zeros((self.C, self.C))

        # Realiza predições para cada amostra
        for t in range(N):
            x_t = X_test_with_bias[:, t:t+1]
            u_t = w_estim.T@x_t

            # Obtém classe predita usando argmax após aplicar função de sinal
            y_t = np.argmax(np.array(
                list(map(ModelPS._sign, u_t))
            ).reshape(self.C, 1))

            # Obtém classe verdadeira
            d_t = np.argmax(Y_test[:, t:t+1])

            # Atualiza matriz de confusão
            self._update_model_precision(int(y_t), int(d_t))

        return self.matriz_conf, self._show_metrics(), self.learning_hist
