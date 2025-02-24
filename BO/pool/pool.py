import numpy as np
from mvlearn.embed import GCCA
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.image as tf_img
from BO.util.util import get_padrao, NOME_PROCESSO
from BO.autoencoder.autoencoder import Autoencoder
from BO.base.base import Base


class Pool:
    def __init__(self, qtd_autoencoders=None, base=None, input_shape=None, tipo_custo_offline=None,
                 tipo_custo_online=None, modelagem=None, diretorio=None):
        """
        Classe Responsável pela criação de Pool de Autoencoders.
        :param qtd_autoencoders: Quantidade de autoencoders que serão criados
        :param base: Base utilizada pelo pool
        :param input_shape: Tamanho da entrada
        :param tipo_custo_offline: Tipo de custo offline
        :param tipo_custo_online: Tipo de custo online
        :param modelagem: Modelagem dos autoencoders do pool
        """
        self.base = base
        self.diretorio = diretorio if diretorio is not None else f"AUTOENCODERS/{get_padrao('BASE.DIRETORIO_TREINO')}/{get_padrao('POOL.MODELAGEM')}"

        self.modelagem = modelagem if modelagem is not None else get_padrao('POOL.MODELAGEM')
        self.input_shape = input_shape if input_shape is not None else get_padrao('BASE.INPUT_SHAPE')

        self.qtd_autoencoders = qtd_autoencoders if qtd_autoencoders is not None else get_padrao('POOL.QUANTIDADE')
        self.tipo_custo_offline = tipo_custo_offline if tipo_custo_offline is not None else get_padrao(
            'POOL.FUNCAO_CUSTO_OFFLINE')
        self.tipo_custo_online = tipo_custo_online if tipo_custo_online is not None else get_padrao(
            'POOL.FUNCAO_CUSTO_ONLINE')

        self.imagens_reconstrucao = []
        self.pool = []
        self.pool_filtrado = []
        self.qtd_erros = 0

    def limpar(self):
        """
        Função responsável por limpar os pools
        :return: Status de limpeza (Sempre True)
        """
        self.pool = []
        self.pool_filtrado = []

        return True

    def criar(self):
        """
        Função responsável pela criação de pool de autoencoders
        :return: Pool de autoencoders criados
        """
        self.carregar_imagens_reconstrucao()
        self.limpar()
        contador = 0
        self.qtd_erros = 0
        while contador < self.qtd_autoencoders:
            if 1==1:#contador > 14:
        #for aec in range(self.qtd_autoencoders):
                aec = contador
                if get_padrao('DEBUG'):
                    print(f'Criando Autoencoder {aec}, erros: {self.qtd_erros}')

                autoencoder = Autoencoder(id=aec, input_shape=self.input_shape, base=self.base, modelagem=self.modelagem)
                autoencoder.criar()
                if self.aplicar_funcao_custo_online(autoencoder=autoencoder):
                    autoencoder.salvar()
                    contador += 1
                else:
                    self.qtd_erros += 1
            else:
                contador += 1


        if get_padrao('POOL.IS_FINNETUNING'):
          self.aplicar_finetunning()

        return self.pool

    def carregar_pool(self, tipo='autoencoder'):
        """
        Função responsável por carregar o pool de autoencoders
        :return: Status de carregamento (Sempre true)
        """
        self.limpar()
        for aec in range(self.qtd_autoencoders):
            if get_padrao('DEBUG'):
                print(f'Carregando {tipo} {aec}')
            if get_padrao('POOL.IS_FINETUNNING'):
                diretorio = f"{self.diretorio}/FINETUNNING/{get_padrao('BASE.DIRETORIO_FINETUNNING')}"
            else:
                diretorio = self.diretorio
            self.pool.append(Autoencoder(id=aec).carregar_model(json_path=f'{self.diretorio}/{str(aec).zfill(3)}/{tipo}.json',
                                                                weights_path=f'{diretorio}/{str(aec).zfill(3)}/{tipo}.weights.h5', tipo=tipo))

        return True

    def aplicar_funcao_custo_online(self, autoencoder=None):
        """
        Função que aplica a punição nos autoencoders para diminuir a quantidade na medida que vai criando os autoencoders
        """
        print('funcao custo online')
        if self.tipo_custo_online == 'SSIM':
            resultado = self.aplicar_ssim_online(autoencoder=autoencoder)
        else:
            resultado = True

        return resultado

    def aplicar_ssim_online(self, autoencoder=None):
        contador = 0
        imagens_reconstruidas = autoencoder.autoencoder.predict(self.imagens_reconstrucao)
        soma_ssim = 0
        for img in self.imagens_reconstrucao:
            vlr_ssim = tf_img.ssim(imagens_reconstruidas[contador], img, max_val=1.0).numpy()
            print(vlr_ssim)
            soma_ssim += vlr_ssim
            contador += 1

        fig, axes = plt.subplots(len(self.imagens_reconstrucao), 2, figsize=(25, 25))
        c = 0
        for i in self.imagens_reconstrucao:
            axes[c, 0].imshow(i)
            axes[c, 0].set_title(f'Original')
            axes[c, 0].axis('off')
            c += 1

        c = 0
        for i in imagens_reconstruidas:
            axes[c, 1].imshow(i)
            axes[c, 1].set_title(f'Refeita')
            axes[c, 1].axis('off')
            c += 1


        #plt.show()

        print(round(soma_ssim/len(self.imagens_reconstrucao),2), get_padrao('POOL.VALOR_CUSTO_THRESHOLD_ONLINE'))
        return round(soma_ssim/len(self.imagens_reconstrucao),2) > get_padrao('POOL.VALOR_CUSTO_THRESHOLD_ONLINE')


    def verificar_reconstrucao(self, predicoes=None):
        """
        Função que verifica se as imagens conseguiram ser, ou não, reconstruidas.
        Se toda a matrix de prediçào for 0, irá dar um erro ao utilizar o GCCA, pois tentará inverter uma matrix zerada.
        :param predicoes: Lista de predições de um autoencoder em cima da base de validação
        :return: Flag que valida se o enconder convergiu ou não
        """
        is_reconstrucao = True
        for p in predicoes:
            if np.sum(p) == 0:
                is_reconstrucao = False

        return is_reconstrucao

    def aplicar_funcao_custo_offline(self):
        """
        Função que aplica a punição nos autoencoders para diminuir a quantidade, após a criação de todos
        :return: Status de aplicação (Sempre true)
        """

        lista_modelos, pool_novo = [], []
        self.carregar_imagens_reconstrucao()

        for modelo in self.pool:
            if get_padrao('DEBUG'):
                print(f'Atualizando Encoder {modelo.id}')
            predicoes = modelo.encoder.predict(self.imagens_reconstrucao)

            is_reconstrucao = self.verificar_reconstrucao(predicoes=predicoes)

            if is_reconstrucao:
                pool_novo.append(modelo)
                lista_modelos.append(predicoes)

        self.pool = pool_novo

        if self.tipo_custo_offline == 'GCCA':
            resultado, encoders_filtrados = self.aplicar_gcca_offline(lista_modelos=lista_modelos)
        else:
            resultado, encoders_filtrados = lista_modelos, [x.id for x in self.pool]

        self.pool_filtrado = []
        for p in self.pool:
            if p.id in encoders_filtrados:
                self.pool_filtrado.append(p)

        with open(f"RESULTADOS/{NOME_PROCESSO}/POOL/configuracoes.txt", "w", encoding='utf-8') as f:
            f.write(f'Tamanho Pool Original: {len(self.pool)} \n')
            f.write(f'Tamanho Pool após filtro de função offline: {len(self.pool_filtrado)} \n')
            f.write(f'Quantidade de autoencoders filtrados pela função online: {self.qtd_erros}')

        self.salvar_grafico_pool(resultado=resultado, encoders_filtrados=encoders_filtrados)

        return True

    def aplicar_gcca_offline(self, lista_modelos=None):
        """
        Função que aplica o método GCCA para encontrar autoencoders similares
        :param lista_modelos: Lista de modelos para o GCCA
        :return: Lista de modelos aplicado o GCCA, Lista de autoencoder que sobraram após a aplicação do GCCA
        """
        self.pool_filtrado = []
        threshold_similaridade = get_padrao('POOL.VALOR_CUSTO_THRESHOLD_OFFLINE')

        gcca = GCCA(n_components=2)
        gcca.fit(lista_modelos)
        resultado_geral = gcca.transform(lista_modelos)

        encoders_similares = []
        matrix_correlacao = np.corrcoef([embedding.flatten() for embedding in resultado_geral])

        for i in range(len(matrix_correlacao)):
            for j in range(i + 1, len(matrix_correlacao)):
                if matrix_correlacao[i, j] >= threshold_similaridade:
                    encoders_similares.append((i, j))

        encoders_filtrados = list(range(len(resultado_geral)))
        for i, j in encoders_similares:
            if j in encoders_filtrados:
                encoders_filtrados.remove(j)

        return resultado_geral, encoders_filtrados

    def carregar_imagens_reconstrucao(self, qtd_imagens_reconstrucao=None):
        """
        Função que carrega as imagens para usar na reconstrução e debug dos dados
        :param qtd_imagens_reconstrucao: Quantidade de imagens para reconstrução
        :return: Lista de imagens para reconstrução
        """
        if qtd_imagens_reconstrucao is None:
            qtd_imagens_reconstrucao = len(self.base.x_test)

        self.imagens_reconstrucao = self.base.x_test[:qtd_imagens_reconstrucao]

        return self.imagens_reconstrucao

    def aplicar_finetunning(self):
        base_finetunning = Base(is_normalizar=True, tipo='labeled', is_base_separada=get_padrao('BASE.IS_DIRETORIO_FINETUNNING_SEPARADO'), diretorio=f"BASE/{get_padrao('BASE.DIRETORIO_FINETUNNING')}")
        _ , _ = base_finetunning.carregar()
        _ = base_finetunning.split_base_validacao()
        x_fine = tf.reshape(base_finetunning.x_train, (-1,) + base_finetunning.x_train[0].shape)
        x_val = tf.reshape(base_finetunning.x_val, (-1,) + base_finetunning.x_val[0].shape)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        for aec in self.pool:
            print(f"FINETUNNING ENCODER: {aec.id}")
            latent_fine = aec.encoder.predict(x_fine)
            latent_val = aec.encoder.predict(x_val)
            aec.encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
            historico = aec.encoder.fit(x_fine, latent_fine, validation_data=(x_val, latent_val), epochs=get_padrao('POOL.FINETUNNING_QTD_EPOCAS'), batch_size=32, shuffle=True, callbacks=[early_stopping])
            nm_diretorio = f"{self.diretorio}/FINETUNNING/{get_padrao('BASE.DIRETORIO_FINETUNNING')}/{str(aec.id).zfill(3)}"
            os.makedirs(nm_diretorio)
            aec.encoder.save_weights(f"{nm_diretorio}/encoder.weights.h5")

            train_loss = historico.history["loss"]
            val_loss = historico.history["val_loss"]  # Only if using validation data

            plt.figure(figsize=(8, 6))
            plt.plot(train_loss, label="Train Loss", color="blue")
            plt.plot(val_loss, label="Validation Loss", color="red", linestyle="dashed")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Autoencoder Loss")
            plt.legend()
            plt.grid()

            plt.savefig(f"{nm_diretorio}/loss.png", bbox_inches="tight")

    def aplicar_finetuning_old(self, x_target=None):
        """
        Essa função aplica o finetuning no pool de autoencoders de uma base alvo e salva o encoder em formato .npy
        :param x_target: Base alvo do finetuning
        :return: Lista de
        """
        resultados = []
        x_target = tf.reshape(x_target, (-1,) + x_target[0].shape)
        for aec in self.pool:
            if get_padrao('DEBUG'):
                print(f'Fazendo predict do encoder {aec.id}')
            resultado = aec.encoder.predict(x_target)
            resultados.append(resultado)
            np.save(f"{self.diretorio}/encoder_{str(aec.id).zfill(3)}", resultado)

        return resultados

    def visualizar_reconstrucao(self, qtd_imagens_reconstrucao=None):
        """
        Função utilizada para visualizar a reconstrução das imagens do pool de autoencoders
        :return: Status de visualização (Sempre true)
        """
        self.carregar_pool(tipo='autoencoder')

        self.carregar_imagens_reconstrucao(qtd_imagens_reconstrucao)

        fig, axes = plt.subplots(self.qtd_autoencoders + 1, qtd_imagens_reconstrucao, figsize=(50, 50))

        for i in range(0, len(self.imagens_reconstrucao)):
            axes[0, i].imshow(self.imagens_reconstrucao[i])
            axes[0, i].set_title(f'Original {i}')
            axes[0, i].axis('off')

        for aec in self.pool:
            imagens_reconstruidas = aec.autoencoder.predict(self.imagens_reconstrucao)
            for j in range(0, qtd_imagens_reconstrucao):
                axes[aec.id + 1, j].imshow(imagens_reconstruidas[j])
                axes[aec.id + 1, j].set_title(f'Reconstruida {j} (AEC {aec.id})')
                axes[aec.id + 1, j].axis('off')

        #plt.show()

        return True

    def salvar_grafico_pool(self, resultado=None, encoders_filtrados=None):
        """
        Função utilizada para printar o resultado dos encoders após filtro do GCCA
        :param resultado: Modelos após a aplicação da base de reconstrução
        :param encoders_filtrados: Lista de ids de encoders filtrados
        :return: Status de plot (Sempre True)
        """
        pca = PCA(n_components=2)
        model_2d = pca.fit_transform([embedding.flatten() for embedding in resultado])

        plt.figure(figsize=(20, 18))
        plt.scatter(model_2d[:, 0], model_2d[:, 1], c=np.arange(len(self.pool)), cmap='viridis', s=100)
        contador = 0
        for aec in self.pool:
            plt.text(model_2d[contador, 0] + 0.01, model_2d[contador, 1] + 0.01, f'Model {aec.id}', fontsize=12,
                     color='#000000' if aec.id in encoders_filtrados else '#FF0000')
            contador += 1

        plt.title('Representação dos Modelos')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.grid(True)
        plt.savefig(f'RESULTADOS/{NOME_PROCESSO}/POOL/modelos.png')

        return True
