from tensorflow.python.keras import layers, models
from keras._tf_keras.keras.models import model_from_json

from BO.autoencoder.configuracao import AutoencoderConfiguracao
from BO.util.util import get_padrao


class Autoencoder(AutoencoderConfiguracao):
    def __init__(self, modelagem=None, base=None, id=None, input_shape=None):
        """
        Classe padrão de criação de autoencoder
        :param modelagem: Modelagem do Autoencoder
        :param base: Base utilizada pelo autoencoder
        :param id: Código de identificação do autoencoder
        :param input_shape: Tamanho das imagens de entrada
        """
        super().__init__(modelagem=modelagem, input_shape=input_shape)
        self.base = base
        self.id = id

    def salvar(self):
        """
        Função utilizada para salvar o autoencoder, encoder e informações do encoder em arquivos
        :return: Status de Salvamento (Sempre True)
        """
        if get_padrao('DEBUG'):
            print(f'Salvando autoencoder {self.id}')

        nm_arquivo = f"AUTOENCODERS/{get_padrao('BASE.DIRETORIO_TREINO')}/{get_padrao('POOL.MODELAGEM')}/autoencoder_{str(self.id).zfill(3)}"
        with open(f"{nm_arquivo}.json", "w") as json_file:
            json_file.write(self.autoencoder.to_json())

        self.autoencoder.save_weights(f"{nm_arquivo}.weights.h5")

        nm_arquivo = f"AUTOENCODERS/{get_padrao('BASE.DIRETORIO_TREINO')}/{get_padrao('POOL.MODELAGEM')}/encoder_{str(self.id).zfill(3)}"
        with open(f"{nm_arquivo}.json", "w") as json_file:
            json_file.write(self.encoder.to_json())

        self.encoder.save_weights(f"{nm_arquivo}.weights.h5")

        with open(f"{nm_arquivo}.txt", 'w') as f:
            f.write(f'AUTOENCODER {self.id}\n')
            f.write(f'SEED: {str(self.seed)}\n')
            f.write(f'LATENTE: {str(self.latente)}\n')
            f.write(f'NR LAYERS: {str(self.nr_layers)}\n')
            f.write(f'FILTROS: {str(self.filtros)}\n')
            f.write(f'KERNEL SIZE: {str(self.kernel_size)}\n')
            f.write(f'ACTIVATION: {str(self.activation)}\n')
            f.write(f'STRIDES: {str(self.strides)}\n')
            f.write(f'PADDING: {str(self.padding)}\n')
            f.write(f'KERNEL INITIALIZER: {str(self.kernel_initializer)}\n')
            f.write(f'OUTPUT ACTIVATION: {str(self.output_activation)}\n')
            f.write(f'QTD EPOCAS: {str(self.qtd_epocas)}\n')

        return True

    def criar(self):
        """
        Função que cria um autoencoder, onde cada etapa é uma função separada
        :return: Autoencoder criado
        """
        self.atualizar_modelagem()

        self.criar_encoder()

        self.criar_decoder()

        self.autoencoder = models.Sequential([self.encoder, self.decoder])

        if get_padrao('DEBUG'):
            self.autoencoder.summary()

        self.treinar()

        self.salvar()

        return self

    def treinar(self):
        """
        Função responsável pelo treinamento do autoencoder
        :return: Status de treinamento (Sempre True)
        """
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.fit(self.base.x_train, self.base.x_train, epochs=self.qtd_epocas, batch_size=64, shuffle=True,
                             validation_data=(self.base.x_test, self.base.x_test))

        return True

    def calcular_saida_encoder(self):
        """
        Função que calcula a saída do encoder para reconstrução exata no decoder
        :return: Altura e largura da saída do encoder
        """
        controle, qtd_layers, valores_strides = self.input_shape[0], 0, []
        for stride in self.strides:
            controle = controle / stride

        return int(controle), int(controle)

    def criar_encoder(self):
        """
        Função responsável em criar a primeira parte do autoencoder, o Encoder.
        Ele começa criando um sequencial, onde o Input é o shape de entrada do Autneocder.
        Após isso são inseridas as camadas de convolução, baseado na arquitetura do autoencoder.
        Por fim, é adicionado a camada do vetor latente, adicionando uma camada dense, do tamanho
        de vetor latente.
        :return: Encoder criado
        """
        self.encoder = models.Sequential()
        self.encoder.add(layers.InputLayer(shape=self.input_shape))
        for camada in range(0, self.nr_layers):
            self.encoder.add(
                layers.Conv2D(filters=self.filtros[camada], kernel_size=self.kernel_size, activation=self.activation,
                              strides=self.strides[camada], padding=self.padding,
                              kernel_initializer=self.kernel_initializer))

        self.encoder.add(layers.Flatten())
        self.encoder.add(layers.Dense(self.latente, activation=self.activation))

        if get_padrao('DEBUG'):
            self.encoder.summary()

        return self.encoder

    def criar_decoder(self):
        """
        Função responsável em criar a segunda parte do autoencoder, o Decoder.
        Ele começa calculando qual é a altura e largura da primeira etapa, para ser possível reconstruir fielmente o decoder,
        baseado no encoder.
        É adicionado uma camada inicial com a altura e largura calculada, e depois adicionado as camadas de convolução transpose,
        com as mesmas configurações do encoder.

        Por fim, é criado a cama de saída, com filtro de 1
        :return: Decoder criado
        """
        altura, largura = self.calcular_saida_encoder()
        self.decoder = models.Sequential()
        self.decoder.add(layers.Input(shape=(self.latente,)))
        self.decoder.add(
            layers.Dense(units=self.filtros[self.nr_layers - 1] * altura * largura, activation=self.activation))
        self.decoder.add(layers.Reshape((altura, largura, self.filtros[self.nr_layers - 1])))

        for camada in range(self.nr_layers - 1, -1, -1):
            self.decoder.add(
                layers.Conv2DTranspose(filters=self.filtros[camada], kernel_size=self.kernel_size,
                                       strides=self.strides[::-1][camada],
                                       activation=self.activation, padding=self.padding,
                                       kernel_initializer=self.kernel_initializer))

        self.decoder.add(layers.Conv2D(filters=1, kernel_size=self.kernel_size, activation=self.output_activation,
                                       padding=self.padding))

        if get_padrao('DEBUG'):
            self.decoder.summary()

        return self.decoder

    def carregar_model(self, json_path=None, weights_path=None, tipo='autoencoder'):
        """
        Função de carregar o autoencoder por meio de um json
        :param json_path: Path do json do modelo
        :param weights_path: Path do arquivo de pesos do modelo
        :param tipo: Tipo do arquivo (Autoencoder ou encoder)
        :return: Modelo carregado
        """
        arquivo_json = open(json_path, 'r')
        model_json_carregado = arquivo_json.read()
        arquivo_json.close()
        model_carregado = model_from_json(model_json_carregado)
        model_carregado.load_weights(weights_path, skip_mismatch=True)
        if tipo == 'encoder':
            self.encoder = model_carregado
        else:
            self.autoencoder = model_carregado

        return self
