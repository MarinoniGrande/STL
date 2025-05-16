# from tensorflow.python.keras import layers, models
# from tensorflow.python.keras.models import model_from_json
import io
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.saving import register_keras_serializable

from BO.autoencoder.configuracao import AutoencoderConfiguracao
from BO.util.util import get_padrao, NOME_PROCESSO


@register_keras_serializable()
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


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
        self.processo = NOME_PROCESSO
        self.diretorio = f"AUTOENCODERS/{self.processo}/{get_padrao('POOL.MODELAGEM')}/{str(self.id).zfill(3)}"

    def salvar(self):
        """
        Função utilizada para salvar o autoencoder, encoder e informações do encoder em arquivos
        :return: Status de Salvamento (Sempre True)
        """
        diretorio = f"AUTOENCODERS/{self.processo}/{get_padrao('POOL.MODELAGEM')}/{str(self.id).zfill(3)}"
        os.makedirs(diretorio, exist_ok=True)

        if get_padrao('DEBUG'):
            print(f'Salvando autoencoder {self.id}')

        nm_arquivo = f'{diretorio}/autoencoder'
        with open(f"{nm_arquivo}.json", "w") as json_file:
            json_file.write(self.autoencoder.to_json())

        self.autoencoder.save_weights(f"{nm_arquivo}.weights.h5")

        summary_io = io.StringIO()
        summary_io.write("========== ENCODER ==========\n")
        summary_io.write("========== INPUT SHAPE ==========\n")
        summary_io.write(f"{str(self.encoder.input_shape)}\n")
        self.encoder.summary(print_fn=lambda x: summary_io.write(x + "\n"))
        summary_io.write("========== DECODER ==========\n")
        self.decoder.summary(print_fn=lambda x: summary_io.write(x + "\n"))
        summary_io.write("========== AUTOENCODER ==========\n")
        self.autoencoder.summary(print_fn=lambda x: summary_io.write(x + "\n"))
        summary_text = summary_io.getvalue()
        summary_io.close()
        lines = summary_text.split("\n")
        with open(f"{diretorio}/sumarios.txt", "w", encoding='utf-8') as f:
            for l in lines:
                f.write(l + '\n')

        nm_arquivo = f"{diretorio}/encoder"
        with open(f"{nm_arquivo}.json", "w") as json_file:
            json_file.write(self.encoder.to_json())

        self.encoder.save_weights(f"{nm_arquivo}.weights.h5")

        with open(f"{nm_arquivo}.txt", 'w', encoding='utf-8') as f:
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
        self.criar_diretorio()

        self.atualizar_modelagem()

        self.criar_encoder()

        self.criar_decoder()

        self.autoencoder = models.Sequential([self.encoder, self.decoder])

        self.treinar()

        # if get_padrao('DEBUG'):
        #     self.autoencoder.summary()

        #self.salvar()

        return self

    def criar_diretorio(self):
        """
        Função responsável por criar o diretório do autoencoder a ser gerado
        :return: Status de criação (Sempre true)
        """
        os.makedirs(self.diretorio, exist_ok=True)

    def treinar(self):
        """
        Função responsável pelo treinamento do autoencoder
        :return: Status de treinamento (Sempre True)
        """
        self.autoencoder.compile(optimizer='adam', loss='mse')

        diretorio = f"AUTOENCODERS/{self.processo}/{get_padrao('POOL.MODELAGEM')}/{str(self.id).zfill(3)}"
        os.makedirs(diretorio, exist_ok=True)

        # checkpoint_callback = ModelCheckpoint(
        #     filepath=f'{diretorio}/autoencoder_best_weights.h5',
        #     save_best_only=True,
        #     monitor='val_loss',
        #     mode='min',
        #     verbose=1
        # )

        early_stopping_callback = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            mode='min',  # Minimize validation loss
            patience=10,  # Stop if no improvement for 10 consecutive epochs
            verbose=1,  # Print messages when stopping
            restore_best_weights=True,  # Load the best weights after stopping
        min_delta = 0.0001,  # Requires at least 0.001 improvement
        )

        historico = self.autoencoder.fit(self.base.x_train, self.base.x_train, epochs=self.qtd_epocas, batch_size=16, shuffle=True,
                             validation_data=(self.base.x_test, self.base.x_test), callbacks=[early_stopping_callback])#, verbose=0)

        train_loss = historico.history["loss"]
        val_loss = historico.history["val_loss"]

        plt.figure(figsize=(8, 6))
        plt.plot(train_loss, label="Train Loss", color="blue")
        plt.plot(val_loss, label="Validation Loss", color="red", linestyle="dashed")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Autoencoder Loss")
        plt.legend()
        plt.grid()

        plt.savefig(f"{self.diretorio}/loss.png", bbox_inches="tight")
        with open(f"{self.diretorio}/early_stopping.txt", "w") as file:
            file.write(f"Best epoch: {early_stopping_callback.best_epoch}\n")
            file.write(f"Stopped epoch: {early_stopping_callback.stopped_epoch}\n")
            file.write(f"Train loss: {historico.history.get('loss')}\n")
            file.write(f"Validation loss: {historico.history.get('val_loss')}\n")

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

            stride_atual = self.strides[camada]
            if stride_atual > 1:
                self.encoder.add(
                    layers.MaxPooling2D(
                        pool_size=(stride_atual, stride_atual),
                        strides=stride_atual,
                        padding=self.padding
                    )
                )

        self.encoder.add(layers.BatchNormalization())
        self.encoder.add(layers.LeakyReLU(alpha=0.5))
        self.encoder.add(layers.Dropout(rate=0.3))
        self.encoder.add(layers.Flatten())

        self.encoder.add(layers.Dense(self.latente, activation=self.activation, name='vetor_latente'))

        # if get_padrao('DEBUG'):
        #     self.encoder.summary()

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

        input_decoder = self.encoder.layers[-2].input
        reshape = (input_decoder.shape[1], input_decoder.shape[2], input_decoder.shape[3])

        self.decoder = models.Sequential()
        self.decoder.add(layers.InputLayer(shape=(self.latente,)))
        self.decoder.add(layers.Dense(units=reshape[0] * reshape[1] * reshape[2],
                                 activation='relu'))
        self.decoder.add(layers.Reshape(reshape))

        for camada in range(self.nr_layers - 1, -1, -1):

            stride_atual = self.strides[::-1][camada]

            if stride_atual > 1:
                self.decoder.add(
                    layers.UpSampling2D(size=(stride_atual, stride_atual))
                )

            self.decoder.add(
                layers.Conv2DTranspose(filters=self.filtros[camada], kernel_size=self.kernel_size,
                                       strides=self.strides[::-1][camada],
                                       activation=self.activation, padding=self.padding,
                                       kernel_initializer=self.kernel_initializer))

        self.decoder.add(layers.BatchNormalization())
        self.decoder.add(layers.LeakyReLU())

        self.decoder.add(layers.Conv2D(filters=self.input_shape[2], kernel_size=self.kernel_size, activation=self.output_activation,
                                       padding=self.padding))

        # if get_padrao('DEBUG'):
        #     self.decoder.summary()

        return self.decoder

    def criar_encoder_o(self):
        """
        Função responsável em criar a primeira parte do autoencoder, o Encoder.
        Ele começa criando um sequencial, onde o Input é o shape de entrada do Autneocder.
        Após isso são inseridas as camadas de convolução, baseado na arquitetura do autoencoder.
        Por fim, é adicionado a camada do vetor latente, adicionando uma camada dense, do tamanho
        de vetor latente.
        :return: Encoder criado
        """
        self.encoder = models.Sequential()
        self.encoder.add(layers.InputLayer(shape=(96, 96, 1)))
        self.encoder.add(layers.Conv2D(filters=128, padding='same', kernel_size=(3, 3), activation='relu'))
        self.encoder.add(layers.Conv2D(filters=64, padding='same', kernel_size=(3, 3), activation='relu'))
        self.encoder.add(layers.MaxPooling2D((2, 2), padding='same'))
        self.encoder.add(layers.Conv2D(filters=256, padding='same', kernel_size=(3, 3), activation='relu'))
        self.encoder.add(layers.Conv2D(filters=64, padding='same', kernel_size=(3, 3), activation='relu'))
        self.encoder.add(layers.Conv2D(filters=128, padding='same', kernel_size=(3, 3), activation='relu'))
        self.encoder.add(layers.Conv2D(filters=64, padding='same', kernel_size=(3, 3), activation='relu'))
        self.encoder.add(layers.Conv2D(filters=64, padding='same', kernel_size=(3, 3), activation='relu'))
        self.encoder.add(layers.Conv2D(filters=128, padding='same', kernel_size=(3, 3), activation='relu'))

        self.encoder.add(layers.BatchNormalization())
        self.encoder.add(layers.LeakyReLU(alpha=0.5))
        self.encoder.add(layers.Dropout(rate=0.3))
        self.encoder.add(layers.Flatten())

        self.encoder.add(layers.Dense(300, activation='relu', name='vetor_latente'))

        if get_padrao('DEBUG'):
            self.encoder.summary()

        return self.encoder

    def criar_decoder_o(self):
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
        reshape = (self.encoder.layers[-2].input.shape[1], self.encoder.layers[-2].input.shape[2],
                   self.encoder.layers[-2].input.shape[3])
        self.decoder.add(layers.InputLayer(shape=(300,)))
        self.decoder.add(layers.Dense(units=reshape[0] * reshape[1] * reshape[2],
                                      activation='relu'))  # encoder.layers[-1].input.shape[1], activation=activation))
        self.decoder.add(layers.Reshape(reshape))  # encoder.layers[-2].input.shape[1:])))

        self.decoder.add(layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.decoder.add(layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.decoder.add(layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.decoder.add(layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))

        self.decoder.add(layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.decoder.add(layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.decoder.add(
            layers.Conv2DTranspose(filters=64, strides=2, kernel_size=(3, 3), padding='same', activation='relu'))

        self.decoder.add(layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))

        self.decoder.add(layers.BatchNormalization())
        self.decoder.add(layers.LeakyReLU(alpha=0.5))

        self.decoder.add(layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu'))

        if get_padrao('DEBUG'):
            self.decoder.summary()

        return self.decoder

    def criar_decoder_e(self):
        """
        Função responsável em criar a segunda parte do autoencoder, o Decoder.
        Ele começa calculando qual é a altura e largura da primeira etapa, para ser possível reconstruir fielmente o decoder,
        baseado no encoder.
        É adicionado uma camada inicial com a altura e largura calculada, e depois adicionado as camadas de convolução transpose,
        com as mesmas configurações do encoder.

        Por fim, é criado a cama de saída, com filtro de 1
        :return: Decoder criado
        """
        reshape = (self.encoder.layers[-2].input.shape[1], self.encoder.layers[-2].input.shape[2],
                   self.encoder.layers[-2].input.shape[3])
        self.decoder = models.Sequential()
        self.decoder.add(layers.InputLayer(shape=(300,)))
        self.decoder.add(layers.Dense(units=reshape[0] * reshape[1] * reshape[2], activation='relu'))#encoder.layers[-1].input.shape[1], activation=activation))
        self.decoder.add(layers.Reshape(reshape))#encoder.layers[-2].input.shape[1:])))

        self.decoder.add(layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.decoder.add(layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.decoder.add(layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.decoder.add(layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))

        self.decoder.add(layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.decoder.add(layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.decoder.add(layers.UpSampling2D((2, 2)))

        self.decoder.add(layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))

        self.decoder.add(layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))

        self.decoder.add(layers.BatchNormalization())
        self.decoder.add(layers.LeakyReLU())

        self.decoder.add(layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), padding='same', activation='relu'))

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
