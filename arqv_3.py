import cv2
import os
import numpy as np

# classepara lidar  com as imagens


class DataFaceRec():
    def _init_(self, path, img_size):
        # Dimensões da imagem
        self.img_size = img_size
        # Path para diretório que segura as pessoas
        self.path = path
        # Path dos diretórios de cada pessoa (20)
        self.path_list_people = self.list_directory()
        self.C = len(self.path_list_people)  # número de classes (20)
        self.N = 32  # número fixo de imagens por pessoa (32)

        # imagens carregadas (NxN) com rotulo associado
        self.img_list_dict: list[dict[str, np.ndarray]] = []

        # Matrizes de imagens e rótulos com a quantidade necessária de espaço
        self.x_matrix = np.ones(
            (img_size[0] * img_size[1] + 1, self.C * self.N))  # +1 para o bias
        self.y_matrix = np.zeros((self.C, self.C * self.N))

        self.get_imgs()  # Carregar imagens com seus rótulos
        self.get_data()  # Preencher as matrizes
        self.normalize_whole_dataset()

    def list_directory(self):
        return [
            os.path.join(self.path, path_dir) for path_dir in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, path_dir))
        ]

    # abra cada imagem e a partir do nome identifique em que categoria está
    def get_imgs(self):
        for i in range(len(self.path_list_people)):
            path = self.path_list_people[i]
            # cada pessoa é uma classe
            label = -np.ones((self.C, 1))
            label[i, 0] = 1
            # path do diretório referente a cada pessoa
            list_path_imgs = [
                os.path.join(path, path_img)
                for path_img in os.listdir(path)
                if os.path.isfile(os.path.join(path, path_img))
            ]

            for img_path in list_path_imgs:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:

                    img_resized = cv2.resize(img, self.img_size)

                    self.img_list_dict.append(
                        {"img": img_resized, "class": label})

    # Tratar dados para estarem na forma apropriada de visualização
    def get_data(self):
        img_idx = 0  # Índice para preencher as matrizes
        for data in self.img_list_dict:
            x = data["img"].flatten()  # Vetor da imagem
            x = np.append(x, -1)  # Adicionar bias
            # Preencher a coluna com o vetor da imagem
            self.x_matrix[:, img_idx] = x
            # Preencher os rótulos
            self.y_matrix[:, img_idx] = data["class"].flatten()
            img_idx += 1

    def normalize_whole_dataset(self):
        # Excluindo a última linha (bias)
        x_data = self.x_matrix[:-1, :]

        # Calculando min e max de todos os dados
        min_val = np.min(x_data)
        max_val = np.max(x_data)

        # Normaliza os valores para o intervalo [0, 1]
        x_data_normalized = 2 * (x_data - min_val) / (max_val - min_val) - 1

        # Atualiza a matriz de dados normalizada
        self.x_matrix[:-1, :] = x_data_normalized

    def MonteCarlo(self):
        mul = self.N * self.C
        index = np.random.permutation(mul)
        division = int(0.8 * mul)
        return (self.x_matrix[:, index[:division]], self.y_matrix[:, index[:division]]), (self.x_matrix[:, index[division:]], self.y_matrix[:, index[division:]])
