import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from fastai.vision.all import TensorImage, Rotate, Zoom, RandomResizedCropGPU, Tensor
from BO.util.util import get_padrao


class Base:
    def __init__(self, diretorio=None, input_shape=None, is_normalizar=False, tipo='unlabeled', labels=None,
                 is_base_separada=False, is_augmentation = False):
        """
        Classe focada em carregar bases de dados de uma forma genérica.
        :param diretorio: Diretório principal
        :param input_shape: Tamanho de cada arquivo
        :param is_normalizar: Flag para verificar se a base deve ser ou não normalizada
        :param tipo: Tipo da base. Se ela for labeled, irá salvar os labels de cada imagem baseado no diretório logo acima deles. Se não, não salva os labels
        :param labels: Lista de labels utilizada para bases labeled
        :param is_base_separada: Flag para verificar se a base já vem separada ou não
        """
        self.diretorio = diretorio if diretorio is not None else f"BASE/{get_padrao('BASE.DIRETORIO_TREINO')}"
        self.input_shape = input_shape if input_shape is not None else get_padrao('BASE.INPUT_SHAPE')
        self.labels = labels if labels is not None else get_padrao('BASE.LABELS')

        self.is_normalizar = is_normalizar
        self.is_augmentation = is_augmentation
        self.is_base_separada = is_base_separada
        self.tipo = tipo

        self.x, self.y = [], []

        self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val = [], [], [], [], [], []

        self.x_fine, self.y_fine = [], []

    def percorrer_diretorio(self, diretorio=None):
        """
        Função recursiva que percorre o diretório de forma genérica. Se for arquivo, ele le o arquivo, e se for um diretório, irá chamar ela mesma
        :param diretorio: Diretório a ser lido
        :return: Status de leitura (Sempre True)
        """
        for dado in os.listdir(diretorio):
            if '.' in dado:
                arquivo = dado
                label = diretorio.split("/")[-1]

                arquivo_tratado = self.tratar_arquivo(diretorio + '/' + arquivo)

                self.x.append(arquivo_tratado)

                if self.is_base_separada:
                    if 'train/' in diretorio.lower():
                        self.x_train.append(arquivo_tratado)
                    else:
                        self.x_test.append(arquivo_tratado)

                if self.tipo == 'labeled':
                    resultado = self.labels.index(label)
                    self.y.append(resultado)
                    if self.is_base_separada:
                        if 'train/' in diretorio.lower():
                            self.y_train.append(resultado)
                        else:
                            self.y_test.append(resultado)

            else:
                self.percorrer_diretorio(diretorio + '/' + dado)

        return True

    def limpar_bases(self):
        """
        Função que serve para limpar os dados da classe
        :return: Status de limpeza (Sempre True)
        """
        self.x, self.y = [], []

        self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val = [], [], [], [], [], []

        return True

    def carregar(self, perc_teste=0.3, is_split_validacao=False):
        """
        Função que irá carregar a base de dados passada.
        :param perc_teste: Percentual da base que vai ser splitada para teste. Se não passar nada, por padrão irá ser 30%
        :param is_split_validacao: Flag que indica se é necessário fazer o split da base para validação
        :return: Tupla de base de dados e seus resultados
        """

        if get_padrao('DEBUG'):
            print(f'Carregando base {self.diretorio}')

        if get_padrao('DEBUG'):
            print(f'Iniciando varredura no diretório {self.diretorio}')

        self.limpar_bases()

        self.percorrer_diretorio(diretorio=self.diretorio)
        self.y = np.array(self.y)

        if self.is_normalizar:
            self.normalizar_base()

        if not self.is_base_separada:
            self.split_base(perc_teste=perc_teste)

        if is_split_validacao:
            self.split_base_validacao()

        if self.is_augmentation:
            self.augmentar_base()

        if get_padrao('DEBUG'):
            self.visualizar_tamanhos()

        return self.x, self.y

    def augmentar_base(self, qtd_augmentar=2):
        if len(self.x_train) > 0:
            base_augmentar = self.x_train
            resultado_augmentar = self.y_train
        else:
            base_augmentar = self.x
            resultado_augmentar = self.y

        x_tensor = [torch.tensor(img).permute(2, 0, 1) for img in base_augmentar]
        tensor_stack = torch.stack(x_tensor)
        size = (self.input_shape[0], self.input_shape[1])

        tensor_batch = TensorImage(tensor_stack)

        lista_augmentation = []
        resultado_augmentation = []
        for i in range(1, qtd_augmentar + 1):
            rot = Rotate()
            zm = Zoom()
            rsc = RandomResizedCropGPU(size, min_scale=0.75)
            dummy_labels = torch.zeros(len(tensor_batch)).long()

            tensor_batch = rot(tensor_batch)
            tensor_batch = zm(tensor_batch)
            tensor_batch, _ = rsc((tensor_batch, dummy_labels))
            lista_augmentation += list(tensor_batch)
            resultado_augmentation += list(resultado_augmentar)

        tensor_batch_concatenado = torch.cat((tensor_stack, torch.stack(lista_augmentation)), dim=0)

        dados_finais = np.array([i.permute(1, 2, 0) for i in tensor_batch_concatenado])

        if len(self.x_train) > 0:
            self.x_train = np.array(dados_finais)
            self.y_train = np.array(list(resultado_augmentar) + resultado_augmentation)
        else:
            self.x = np.array(dados_finais)
            self.y  = np.array(list(resultado_augmentar) + resultado_augmentation)

        return self.x_train

    def tratar_arquivo(self, arquivo=None):
        """
        Função que recebe um arquivo e trata ele para deixar todos no mesmo padrão
        :param arquivo: Path completa do arquivo
        :return: Objeto Arquivo tratado
        """
        if self.input_shape[2] == 1:
            arq = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)
        else:
            arq = cv2.imread(arquivo)
            arq = cv2.cvtColor(arq, cv2.COLOR_BGR2RGB)

        arq = cv2.resize(arq, (self.input_shape[0], self.input_shape[1]), interpolation=cv2.INTER_AREA)
        arq = np.reshape(arq, self.input_shape)
        return arq

    def visualizar_tamanhos(self):
        """
        Função responsável por visualizar o tamanho de cada dado na base.
        :return: Status da visualização (Sempre true)
        """
        print(f'Tamanho da base (X): {len(self.x)}')
        print(f'Tamanho dos resultados (Y): {len(self.y)}')

        print(f'Tamanho da base de treino (X train): {len(self.x_train)}')
        print(f'Tamanho dos resultados de treino (Y train): {len(self.y_train)}')

        print(f'Tamanho da base de testes (X test): {len(self.x_test)}')
        print(f'Tamanho dos resultados de testes (Y test): {len(self.y_test)}')

        print(f'Tamanho da base de validação (X val): {len(self.x_val)}')
        print(f'Tamanho dos resultados de validação (Y val): {len(self.y_val)}')

    def normalizar_base(self):
        """
        Função responsável para normalizar a base
        :return: Base normalizada (0 a 1)
        """
        self.x = np.array(self.x).astype('float32') / 255.0
        return self.x

    def split_base_validacao(self):
        """
        Função responsável por separar a base treino em validação
        :return: Base de treino e de validação
        """

        random_state = random.randint(1, 100) if get_padrao('SEEDS.IS_ALEATORIO') else get_padrao('SEEDS.BASE')

        if self.tipo == 'labeled':
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=random_state)

        else:
            self.x_train, self.x_val = train_test_split(self.x, test_size=0.2, random_state=random_state)

        return self.x_train, self.x_val

    def split_base_finetunning(self):
        """
        Função responsável por separar a base treino em finetunning
        :return: Base de treino e de validação
        """
        random_state = random.randint(1, 100) if get_padrao('SEEDS.IS_ALEATORIO') else get_padrao('SEEDS.BASE')

        if self.tipo == 'labeled':
            self.x_train, self.x_fine, self.y_train, self.y_fine = train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=random_state)

        else:
            self.x_train, self.x_fine = train_test_split(self.x, test_size=0.2, random_state=random_state)

        return self.x_train, self.x_fine

    def split_base(self, perc_teste=0.3):
        """
        Função responsável por fazer o split da base em dados testes e dados treinos.
        :param perc_teste: Percentual da base que vai ser splitada para teste. Se não passar nada, por padrão irá ser 30%
        :return: Tupla de base de dados de treino e resultados de treino
        """
        random_state = random.randint(1, 100) if get_padrao('SEEDS.IS_ALEATORIO') else get_padrao('SEEDS.BASE')

        if self.tipo == 'labeled':
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=perc_teste, random_state=random_state)

        else:
            self.x_train, self.x_test = train_test_split(self.x, test_size=perc_teste, random_state=random_state)

        return self.x_train, self.x_test
