import random
import tensorflow as tf

from BO.util.util import get_padrao, get_valor_aleatorio


class AutoencoderConfiguracao:
    def __init__(self, modelagem=None, input_shape=None):
        """
        Classe de configurações de autoencoders padrão
        :param modelagem: Modelagem do Autoencoder
        :param input_shape: Tamanho da entrada
        """
        self.modelagem = modelagem if modelagem is not None else get_padrao('POOL.MODELAGEM')

        self.input_shape = input_shape
        self.filtros = None
        self.kernel_size = None
        self.activation = None
        self.strides = None
        self.padding = None
        self.kernel_initializer = None
        self.nr_layers = None
        self.output_activation = None
        self.qtd_epocas = None

        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.latente = None
        self.seed = None

    def atualizar_modelagem(self):
        """
        Função para atualizar os dados do autoencoder baseado na sua modelagem (SLA-5-3-2)
        :return: Status de atualização
        """
        # MODELAGEM 'S'
        self.gerar_seed()

        # MODELAGEM 'L'
        self.gerar_latente()

        # MODELAGEM 'A'
        self.gerar_arquitetura()

        return True

    def gerar_seed(self):
        """
        Função que gera a seed do autoencoder baseado na modelagem
        :return: Seed gerada
        """
        if 'S' in self.modelagem.upper():
            self.seed = random.randint(get_padrao('AUTOENCODER.SEED_RANDOM_INI'), get_padrao('AUTOENCODER.SEED_RANDOM_FIM'))
        else:
            self.seed = get_padrao('AUTOENCODER.SEED_PADRAO')

        if get_padrao('SEEDS.IS_ALEATORIO'):
            tf.random.set_seed(self.seed)

        if get_padrao('DEBUG'):
            print(f'Seed: {self.seed}')

        return self.seed

    def gerar_latente(self):
        """
        Função que gera o vetor latente do autoencoder baseado na modelagem
        :return: Vetor latente gerada
        """
        if 'L' in self.modelagem.upper():
            self.latente = random.randint(get_padrao('AUTOENCODER.VETOR_LATENTE_RANDOM_INI'),
                                          get_padrao('AUTOENCODER.VETOR_LATENTE_RANDOM_FIM'))
        else:
            self.latente = get_padrao('AUTOENCODER.VETOR_LATENTE_PADRAO')

        if get_padrao('DEBUG'):
            print(f'Latente: {self.latente}')

        return self.latente

    def gerar_arquitetura(self):
        """
        Função que gera a arquitetura do autoencoder baseado na modelagem
        :return: Classe atualizada
        """
        if 'A' in self.modelagem.upper():
            self.nr_layers, self.filtros, self.strides = self.get_layers_aleatorio()
            self.kernel_size = self.get_kernel_size_aleatorio()
            self.activation = self.get_activation_aleatorio()
            self.padding = self.get_padding_aleatorio()
            self.kernel_initializer = self.get_kernel_initializer_aleatorio()
            self.output_activation = self.get_output_activation_aleatorio()
            self.qtd_epocas = self.get_qtd_epocas_aleatorio()
        else:
            self.nr_layers = get_padrao('AUTOENCODER.NR_LAYERS_PADRAO')
            self.filtros = get_padrao('AUTOENCODER.FILTROS_PADRAO')
            self.kernel_size = tuple(get_padrao('AUTOENCODER.KERNEL_SIZE_PADRAO'))
            self.activation = get_padrao('AUTOENCODER.ACTIVATION_PADRAO')
            self.strides = get_padrao('AUTOENCODER.STRIDES_PADRAO')
            self.padding = get_padrao('AUTOENCODER.PADDING_PADRAO')
            self.kernel_initializer = get_padrao('AUTOENCODER.KERNEL_INITIALIZER_PADRAO')
            self.output_activation = get_padrao('AUTOENCODER.OUTPUT_ACTIVATION_PADRAO')
            self.qtd_epocas = get_padrao('AUTOENCODER.EPOCAS_PADRAO')

        if get_padrao('DEBUG'):
            print(f'Nr. Layers: {self.nr_layers}')
            print(f'Filtros: {self.filtros}')
            print(f'Kernel Size: {self.nr_layers}')
            print(f'Activation: {self.activation}')
            print(f'strides: {self.strides}')
            print(f'Padding: {self.padding}')
            print(f'Kernel Initializer: {self.kernel_initializer}')
            print(f'Output Activation: {self.output_activation}')
            print(f'Qtd. Epocas: {self.qtd_epocas}')

        return self

    def get_layers_aleatorio(self):
        """
        Função que retorna a quantidade de layers e os valores de cada layer
        :return: Número de layeres e valores de layers aleatório
        """

        nr_layers = random.randint(get_padrao('AUTOENCODER.LAYERS_RANDOM_INI'), get_padrao('AUTOENCODER.LAYERS_RANDOM_FIM'))
        valores_layers = []
        for valor in range(0, nr_layers):
            valores_layers.append(get_valor_aleatorio(get_padrao('AUTOENCODER.LAYERS_RANDOM')))

        controle, qtd_layers, valores_strides = self.input_shape[0], 0, []

        for _ in valores_layers:
            if controle == 1:
                break
            if controle % 2 == 0:
                controle = controle / 2
                valores_strides.append(2)
            elif controle % 3 == 0:
                controle = controle / 3
                valores_strides.append(3)
            else:
                break
            qtd_layers += 1

        nr_layers = qtd_layers
        valores_layers = valores_layers[:nr_layers]
        return nr_layers, valores_layers, valores_strides

    def get_kernel_size_aleatorio(self):
        """
        Função que retorna um valor de kernel size aleatorio
        :return: Kernel size aleatório
        """
        valor = get_valor_aleatorio([2, 3])
        return valor, valor

    def get_activation_aleatorio(self):
        """
        Função que retorna um valor de activation aleatorio
        :return: Activation aleatório
        """
        return get_valor_aleatorio(['relu'])

    def get_padding_aleatorio(self):
        """
        Função que retorna um valor de padding aleatorio
        :return: Padding aleatório
        """
        return get_valor_aleatorio(['same'])

    def get_kernel_initializer_aleatorio(self):
        """
        Função que retorna um valor de kernel initializer aleatorio
        :return: Kernel initializer aleatório
        """
        return get_valor_aleatorio(['he_uniform'])

    def get_strides_aleatorio(self):
        """
        Função que retorna um valor de strides aleatorio
        :return: Striders aleatório
        """
        return get_valor_aleatorio([2, 3])

    def get_output_activation_aleatorio(self):
        """
        Função que retorna um valor de output activation aleatorio
        :return: Output activation aleatório
        """
        return get_valor_aleatorio(['linear'])

    def get_qtd_epocas_aleatorio(self):
        """
        Função que retorna a quantidade de épocas aleatorio
        :return: Quantidade de épocas aleatório
        """
        return random.randint(get_padrao('AUTOENCODER.EPOCAS_RANDOM_INI'), get_padrao('AUTOENCODER.EPOCAS_RANDOM_FIM'))
