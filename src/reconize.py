import os
import cv2
from pprint import pprint


class DirFileExplorer:
    '''
        Nessa classe, foco para extrair quais as pastas e os caminhos 
        relativos dos arquivos.
    '''

    def __init__(self, dir_name):
        # Caminho da pasta inicial
        self.path = os.path.relpath(dir_name)

        # Nomes das pastas e os caminhos dos seus arquivos
        self.img_dirs = {d: self.copy_imgs(d) for d in os.listdir(self.path)}

    def copy_imgs(self, dir_name):
        sub_dir = os.path.join(self.path, dir_name)
        images = []
        with os.scandir(sub_dir) as arquivos:
            for arquivo in arquivos:
                if os.path.isfile(arquivo):
                    images.append(os.path.join(sub_dir, arquivo.name))
        return images


class ClassFacial:
    def __init__(self, file_name, witdth, height):
        self.data = DirFileExplorer(file_name)
        self.matrix_dict = self.create_matrix_dict(witdth, height)

    @staticmethod
    def read_and_resize_img(imgs, widht, height):
        matrix_list = []
        for img in imgs:
            img_matrix = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            img_matrix = cv2.resize(img_matrix, (widht, height))
            matrix_list.append(img_matrix)
        return matrix_list

    def create_matrix_dict(self, widht, height):
        return {
            subdir: self.read_and_resize_img(
                self.data.img_dirs[subdir], widht, height)
            for subdir in self.data.img_dirs.keys()
        }


if __name__ == '__main__':
    teste_com_10 = ClassFacial('RecFac', 10, 10)
    teste_com_20 = ClassFacial('RecFac', 20, 20)
    teste_com_30 = ClassFacial('RecFac', 30, 30)
    teste_com_40 = ClassFacial('RecFac', 40, 40)
    teste_com_50 = ClassFacial('RecFac', 50, 50)

    pprint(teste_com_10.matrix_dict)
