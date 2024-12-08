import numpy as np
import matplotlib.pyplot as plt
# FUNÇÃO DE ATIVAÇÃO: DEGRAU BIPOLAR - FUNÇÃO SINAL - SIGN FUNCTION


def sign(u):
    return 1 if u >= 0 else -1


X = np.array([
    [1, 1],
    [0, 1],
    [0, 2],
    [1, 0],
    [2, 2],
    [4, 1.5],
    [1.5, 6],
    [3, 5],
    [3, 3],
    [6, 4]])

Y = np.array([
    [1],
    [1],
    [1],
    [1],
    [1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],])

# Visualização dos dados:
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], s=90,
            marker='*', color='blue', label='Classe +1')
plt.scatter(X[Y[:, 0] == -1, 0], X[Y[:, 0] == -1, 1], s=90,
            marker='s', color='red', label='Classe -1')
plt.legend()
plt.ylim(-0.5, 7)
plt.xlim(-0.5, 7)

# Organização dos dados:
# Passo 1: Organizar os dados de treinamento com a dimensão (p x N)
X = X.T
Y = Y.T
p, N = X.shape

# Passo 2: Adicionar o viés (bias) em cada uma das amostras:
X = np.concatenate((
    -np.ones((1, N)),
    X)
)

# Modelo do Perceptron Simples:
lr = .001  # Definição do hiperparâmetro Taxa de Aprendizado (Learning Rate)

# Inicialização dos parâmetros (pesos sinápticos e limiar de ativação):
w = np.zeros((3, 1))  # todos nulos
# parâmetros aleatórios entre -0.5 e 0.5
w = np.random.random_sample((3, 1))-.5

# plot da reta que representa o modelo do perceptron simples em sua inicialização:
x_axis = np.linspace(-2, 10)
x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
x2 = np.nan_to_num(x2)
plt.plot(x_axis, x2, color='k')


# condição inicial
erro = True
epoca = 0  # inicialização do contador de épocas.
while (erro):
    erro = False
    for t in range(N):
        x_t = X[:, t].reshape(p+1, 1)
        u_t = (w.T@x_t)[0, 0]
        y_t = sign(u_t)
        d_t = float(Y[0, t])
        e_t = d_t - y_t
        w = w + (lr*e_t*x_t)/2
        if (y_t != d_t):
            erro = True
    # plot da reta após o final de cada época
    plt.pause(.01)
    x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
    x2 = np.nan_to_num(x2)
    plt.plot(x_axis, x2, color='orange', alpha=.1)
    epoca += 1
print(w)
print(w.shape)
# fim do treinamento
x2 = -w[1, 0]/w[2, 0]*x_axis + w[0, 0]/w[2, 0]
x2 = np.nan_to_num(x2)
line = plt.plot(x_axis, x2, color='green', linewidth=3)
plt.show()
