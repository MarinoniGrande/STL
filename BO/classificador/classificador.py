from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

from BO.util.util import get_padrao, NOME_PROCESSO
from BO.base.base import Base
from mvlearn.embed import GCCA

from sklearn.decomposition import PCA


class Classificador:
    def __init__(self, pool=None):
        """
        Classe responsável de fazer a classificação das imagens alvo
        :param base: Base de dado alvo
        :param classificador: Classificador utilizado
        :param estimator: Estimator usado no classificador
        """
        self.x = []
        self.classificador = get_padrao('CLASSIFICADOR.CLASSIFICADOR')
        self.estimator = get_padrao('CLASSIFICADOR.ESTIMADOR')

        self.resultado = []

        self.base_teste = Base(is_normalizar=True, tipo='labeled', is_base_separada=get_padrao('BASE.IS_DIRETORIO_ALVO_TESTE_SEPARADO'), diretorio=f"{get_padrao('BASE.DIRETORIO_ALVO_TESTE')}")
        self.base_treino = Base(is_normalizar=True, is_augmentation=True, tipo='labeled', is_base_separada=get_padrao('BASE.IS_DIRETORIO_ALVO_TREINO_SEPARADO'), diretorio=f"{get_padrao('BASE.DIRETORIO_ALVO_TREINO')}")
        self.pool = pool

        self.carregar_bases()

    def carregar_bases(self):

        self.base_treino.carregar(is_split_validacao=True)

        self.base_teste.carregar()

    def get_classificador(self):
        dict_classificadores = {
            'BAGGING': BaggingClassifier(estimator=self.get_estimator(), n_estimators=get_padrao('CLASSIFICADOR.QTD_ESTIMATORS'), random_state=get_padrao('SEEDS.CLASSIFICADOR')),
            'RANDOMFOREST': RandomForestClassifier(n_estimators=get_padrao('CLASSIFICADOR.QTD_ESTIMATORS'), random_state=get_padrao('SEEDS.CLASSIFICADOR')),
            'SVM': SVC(probability=True, random_state=get_padrao('SEEDS.CLASSIFICADOR'))
        }
        return dict_classificadores.get(self.classificador)

    def get_estimator(self):
        dict_estimators = {
            'decisiontree': DecisionTreeClassifier(random_state=get_padrao('SEEDS.CLASSIFICADOR'))
        }
        return dict_estimators.get(self.estimator)

    # Funções de classificação
    def treinar_classificador_old(self, encoder=None, tipo=None, x_test=None, x_train=None, y_train=None, x_val=None, y_val=None,
                              trainable=False):
        x_test = tf.reshape(x_test, (-1,) + x_test[0].shape)
        x_train_flat, y_train_flat = np.array(x_train), tf.keras.utils.to_categorical(y_train, num_classes=2)
        x_val_flat, y_val_flat = np.array(x_val), tf.keras.utils.to_categorical(y_val, num_classes=2)

        if tipo == 'SVM':

            x_train_encoded = encoder.predict(x_train_flat, batch_size=16)
            classificador = SVC(probability=True, random_state=9)
            classificador.fit(x_train_encoded, y_train)

            y_pred_enc = encoder.predict(x_test)
            y_pred = classificador.predict_proba(y_pred_enc)
            y_pred = np.argmax(y_pred, axis=1)

        else:
            print('aqui')
            encoder.trainable = trainable
            x = encoder.outputs[0]
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dense(128, activation='relu')(x)
            output = layers.Dense(2, activation='softmax', name="segundo_dense")(x)

            early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1,
                                                    restore_best_weights=True, min_delta=0.001)

            new_model = models.Model(inputs=encoder.inputs[0], outputs=output)

            # Compile the new model
            new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                              metrics=['accuracy'])

            new_model.fit(x_train_flat, y_train_flat, epochs=100, validation_data=(x_val_flat, y_val_flat),
                          callbacks=[early_stopping_callback])

            y_pred = new_model.predict(x_test)

            y_pred = np.argmax(y_pred, axis=1)

        return y_pred


    def treinar_classificador(self, encoder=None):
        x_test = tf.reshape(self.base_teste.x_test, (-1,) + self.base_teste.x_test[0].shape)
        x_train_flat, y_train_flat = np.array(self.base_treino.x_train), tf.keras.utils.to_categorical(self.base_treino.y_train, num_classes=2)
        x_val_flat, y_val_flat = np.array(self.base_treino.x_val), tf.keras.utils.to_categorical(self.base_treino.y_val, num_classes=2)

        if self.classificador == 'CNN':
            encoder.trainable = get_padrao('CLASSIFICADOR.ESTIMADOR')
            x = encoder.outputs[0]
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dense(128, activation='relu')(x)
            output = layers.Dense(2, activation='softmax', name="segundo_dense")(x)

            early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1,
                                                    restore_best_weights=True, min_delta=0.001)

            new_model = models.Model(inputs=encoder.inputs[0], outputs=output)

            # Compile the new model
            new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                              metrics=['accuracy'])

            new_model.fit(x_train_flat, y_train_flat, epochs=100, validation_data=(x_val_flat, y_val_flat),
                          callbacks=[early_stopping_callback])

            predicoes = new_model.predict(x_test)

            y_pred = np.argmax(predicoes, axis=1)
        else:
            x_train_encoded = encoder.predict(x_train_flat, batch_size=16)
            classificador = self.get_classificador()
            classificador.fit(x_train_encoded, self.base_treino.y_train)

            y_pred_enc = encoder.predict(x_test)
            predicoes = classificador.predict_proba(y_pred_enc)
            y_pred = np.argmax(predicoes, axis=1)

        return y_pred, predicoes

    def carregar_vetores(self, pool=None, imagens_reconstrucao=None):
        lista_predicoes = []
        novo_pool = []
        for enc in pool:
            modelo = enc.encoder
            # print(f'enc {enc.get("nome")}')
            predicoes = modelo.predict(imagens_reconstrucao)

            is_reconstrucao = True
            for p in predicoes:
                if np.sum(p) == 0:
                    is_reconstrucao = False

            if is_reconstrucao:
                novo_pool.append(enc)
                lista_predicoes.append(predicoes)
        return lista_predicoes, novo_pool

    def classificar(self, base_test=None):

        base_treino_cla = self.base_treino

        base_teste_cla = self.base_teste
        nm_diretorio = f'RESULTADOS/{NOME_PROCESSO}/CLASSIFICADOR/'
        os.makedirs(f'{nm_diretorio}', exist_ok=True)

        print('1')
        imagens_reconstrucao = np.array(base_test.x_train[:40]) # alteracao para finetunnig na mesma base
        lista_predicoes, novo_pool = self.carregar_vetores(pool=self.pool.pool, imagens_reconstrucao=imagens_reconstrucao)

        n_components = int(min([l.shape[0] for l in lista_predicoes] + [l.shape[1] for l in lista_predicoes]))
        print('2')
        gcca = GCCA(n_components=n_components - 1)  # `k` must be an integer satisfying `0 < k < min(A.shape)`.
        gcca.fit(lista_predicoes)
        resultado_geral = gcca.transform(lista_predicoes)
        print('3')
        similarity_matrix = np.corrcoef([embedding.flatten() for embedding in resultado_geral])
        # print(similarity_matrix)

        threshold_similaridade = 0.7
        # similarity_matrix = cosine_similarity(encoder_vectors)
        # Optional: plot similarity matrix

        plt.figure(figsize=(18, 9))
        sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm',
                    xticklabels=[f'Enc {novo_pool[i].id}' for i in range(len(resultado_geral))],
                    yticklabels=[f'Enc {novo_pool[i].id}' for i in range(len(resultado_geral))])
        plt.title("Encoder Similarity")

        plt.savefig(f'{nm_diretorio}/similaridade.png')
        #plt.show()
        print('4')
        encoders_similares = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i, j] >= threshold_similaridade:
                    encoders_similares.append((i, j))

        encoders_filtrados = list(range(len(resultado_geral)))
        for i, j in encoders_similares:
            if j in encoders_filtrados:
                encoders_filtrados.remove(j)
        encoders_filtrados = [novo_pool[e].id for e in encoders_filtrados]
        _ = self.plot_tipo(tipo='pca', resultado_geral=resultado_geral, encoders_filtrados=encoders_filtrados,
                             pool=novo_pool, diretorio=nm_diretorio)
        print('5')
        x_test = tf.reshape(base_teste_cla.x_test, (-1,) + base_teste_cla.x_test[0].shape)
        x_train_flat, y_train_flat = np.array(base_treino_cla.x_train), tf.keras.utils.to_categorical(
            base_treino_cla.y_train, num_classes=len(get_padrao('BASE.LABELS')))
        x_val_flat, y_val_flat = np.array(base_treino_cla.x_val), tf.keras.utils.to_categorical(base_treino_cla.y_val,num_classes=len(get_padrao('BASE.LABELS')))

        resultado, resultado_filtro = [], []
        for p in novo_pool:
            print(p.id)
            encoder = p.encoder

            if self.classificador == 'SVC':
                x_train_encoded = encoder.predict(x_train_flat, batch_size=16)
                classificador = SVC(probability=True, random_state=42)
                classificador.fit(x_train_encoded, base_treino_cla.y_train)
                y_pred_enc = encoder.predict(x_test)
                predicoes = classificador.predict_proba(y_pred_enc)
            else:
                if 'TRUE' in self.classificador:
                    encoder.trainable = True
                else:
                    encoder.trainable = False
                x = encoder.outputs[0]
                x = layers.Dense(512, activation='relu')(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.Dense(128, activation='relu')(x)
                output = layers.Dense(len(get_padrao('BASE.LABELS')), activation='softmax', name="segundo_dense")(x)

                early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1,
                                                        restore_best_weights=True, min_delta=0.001)

                new_model = models.Model(inputs=encoder.inputs[0], outputs=output)

                # Compile the new model
                new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

                new_model.fit(x_train_flat, y_train_flat, epochs=150, validation_data=(x_val_flat, y_val_flat),
                              callbacks=[early_stopping_callback])

                predicoes = new_model.predict(x_test)


            y_pred = np.argmax(predicoes, axis=1)
            if p.id in encoders_filtrados:
                resultado_filtro.append(predicoes)
            resultado.append(predicoes)
        print('6')
        prod = np.product(resultado, axis=0)
        lista_produto = [np.argmax(x) for x in prod]

        sum = np.sum(resultado, axis=0)
        lista_soma = [np.argmax(x) for x in sum]

        prod_filtrado = np.product(resultado_filtro, axis=0)
        lista_produto_filtrado = [np.argmax(x) for x in prod_filtrado]

        sum_filtrado = np.sum(resultado_filtro, axis=0)
        lista_soma_filtrado = [np.argmax(x) for x in sum_filtrado]

        with open(f"{nm_diretorio}/RESULTADO.txt", "w") as f:
            f.write(f'TOTAL SOMA: {accuracy_score(base_teste_cla.y_test, lista_soma)}, PRODUTO: {accuracy_score(base_teste_cla.y_test, lista_produto)}\n')
            f.write(f'FILTRADO: SOMA {accuracy_score(base_teste_cla.y_test, lista_soma_filtrado)}, PRODUTO {accuracy_score(base_teste_cla.y_test, lista_produto_filtrado)}\n')
        print('7')
        return True


    def plot_tipo(self, tipo='pca', resultado_geral=None, encoders_filtrados=[], pool=None, diretorio=None):
        pca = PCA(n_components=2)
        model_2d = pca.fit_transform([embedding.flatten() for embedding in resultado_geral])

        plt.figure(figsize=(32, 16))
        plt.scatter(model_2d[:, 0], model_2d[:, 1], c=np.arange(len(pool)), cmap='viridis', s=100)
        contador = 0
        for aec in pool:
            plt.text(model_2d[contador, 0] + 0.01, model_2d[contador, 1] + 0.01, f'Model {aec.id}',
                     fontsize=12,
                     color='#000000' if aec.id in encoders_filtrados else '#FF0000')
            contador += 1

        print(f'Encoders restantes ({len(encoders_filtrados)}): {encoders_filtrados}')
        plt.title(f'Representação dos Modelos {tipo} {len(encoders_filtrados)}): {encoders_filtrados}')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.grid(True)
        plt.savefig(f'{diretorio}/pca.png')
        return True

    def classificar_antigo(self):
        for autoencoder in self.pool.pool:
            y_pred, predicoes = self.treinar_classificador(encoder=autoencoder.encoder)
            self.resultado.append(predicoes)

            missed_indices = np.where(self.base_teste.y_test != y_pred)[0]
            correct_indices = np.where(self.base_teste.y_test == y_pred)[0]

            resultado_acc = accuracy_score(self.base_teste.y_test, y_pred)

            nm_diretorio = f'RESULTADOS/{NOME_PROCESSO}/CLASSIFICADOR/{str(autoencoder.id).zfill(3)}'
            os.makedirs(f'{nm_diretorio}', exist_ok=True)

            plt.figure(figsize=(20, 16))

            shape = get_padrao('BASE.INPUT_SHAPE')[0]
            cor = get_padrao('BASE.INPUT_SHAPE')[2]
            for i in range(8):  # Show 5 missed images
                try:
                    plt.subplot(4, 8, i + 1)
                    plt.imshow(self.base_teste.x_test[missed_indices[i]].reshape(shape, shape, cor), cmap='gray')
                    plt.title(
                        f"Ind: {missed_indices[i]} - True:{self.base_teste.y_test[missed_indices[i]]}, Pred:{y_pred[missed_indices[i]]}")
                except:
                    pass

            for i in range(8):  # Show 5 missed images
                try:
                    plt.subplot(4, 8, i + 9)
                    plt.imshow(self.base_teste.x_test[missed_indices[::-1][i]].reshape(shape, shape, cor), cmap='gray')
                    plt.title(
                        f"Ind: {missed_indices[::-1][i]} - True:{self.base_teste.y_test[missed_indices[::-1][i]]}, Pred:{y_pred[missed_indices[::-1][i]]}")
                except Exception as e:
                    print(e)
                    pass

            for i in range(8):  # Show 5 correctly classified images
                try:
                    plt.subplot(4, 8, i + 17)
                    plt.imshow(self.base_teste.x_test[correct_indices[i]].reshape(shape, shape, cor), cmap='gray')
                    plt.title(
                        f"Ind: {correct_indices[i]} - True: {self.base_teste.y_test[correct_indices[i]]}, Pred:{y_pred[correct_indices[i]]}")
                except:
                    pass

            for i in range(8):  # Show 5 correctly classified images
                try:
                    plt.subplot(4, 8, i + 25)
                    plt.imshow(self.base_teste.x_test[correct_indices[::-1][i]].reshape(shape, shape, cor), cmap='gray')
                    plt.title(
                        f"Ind: {correct_indices[::-1][i]} - True: {self.base_teste.y_test[correct_indices[::-1][i]]}, Pred:{y_pred[correct_indices[::-1][i]]}")
                except:
                    pass

            plt.tight_layout()
            plt.savefig(f"{nm_diretorio}/PREDICOES.png")

            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(self.base_teste.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=get_padrao('BASE.LABELS'),
                        yticklabels=get_padrao('BASE.LABELS'))
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.savefig(f"{nm_diretorio}/MATRIZ_CONFUSAO.png")

            with open(f"{nm_diretorio}/RESULTADO.txt", "w") as f:
                f.write(f'ACC: {resultado_acc}\n')
                f.write('Corretos:')
                f.write(str(list(np.array(correct_indices))) + '\n')
                f.write('Incorretos:')
                f.write(str(list(np.array(missed_indices))) + '\n')

        self.calcular_acuraria()


    def classificar_old(self):
        if self.classificador == 'CNN':
            from tensorflow.keras import layers, models
            for autoencoder in self.pool.pool:
                autoencoder.encoder.trainable = True  # Freeze the encoder layers

                # Add new layers to the encoder
                x = autoencoder.encoder.outputs[0]
                x = layers.Dense(512, activation='relu')(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.Dense(128, activation='relu')(x)
                output = layers.Dense(2, activation='softmax', name="segundo_desnse")(x)  # Output layer with softmax for 2 classes

                # Create the new model
                new_model = models.Model(inputs=autoencoder.encoder.inputs[0], outputs=output)

                # Compile the new model
                new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                # Print the new model summary
                new_model.summary()

                base_teste = tf.reshape(self.base_teste.x_test, (-1,) + self.base_teste.x_test[0].shape)
                x_train_flat = np.array(self.base_treino.x_train)

                y_train_flat = tf.keras.utils.to_categorical(self.base_treino.y_train, num_classes=2)
                new_model.fit(x_train_flat , y_train_flat, epochs=100)

                y_pred = new_model.predict(base_teste)

                labels_resultado = np.argmax(y_pred, axis=1)
                cm = confusion_matrix(self.base_teste.y_test, labels_resultado)

                missed_indices = np.where(self.base_teste.y_test != labels_resultado)[0]
                correct_indices = np.where(self.base_teste.y_test == labels_resultado)[0]

                plt.figure(figsize=(20, 16))
                for i in range(8):  # Show 5 missed images
                    try:
                        plt.subplot(4, 8, i + 1)
                        plt.imshow(self.base_teste.x_test[missed_indices[i]].reshape(200, 200), cmap='gray')
                        plt.title(f"True {i}:{self.base_teste.y_test[missed_indices[i]]}, Pred:{labels_resultado[missed_indices[i]]}")
                    except:
                        pass

                for i in range(8):  # Show 5 missed images
                    try:
                        plt.subplot(4, 8, i + 9)
                        plt.imshow(self.base_teste.x_test[missed_indices[::-1][i]].reshape(200, 200), cmap='gray')
                        plt.title(f"True {i}:{self.base_teste.y_test[missed_indices[::-1][i]]}, Pred:{labels_resultado[missed_indices[::-1][i]]}")
                    except:
                        pass

                for i in range(8):  # Show 5 correctly classified images
                    try:
                        plt.subplot(4, 8, i + 17)
                        plt.imshow(self.base_teste.x_test[correct_indices[i]].reshape(200, 200), cmap='gray')
                        plt.title(f"True {i}: {self.base_teste.y_test[correct_indices[i]]}, Pred:{labels_resultado[correct_indices[i]]}")
                    except:
                        pass

                for i in range(8):  # Show 5 correctly classified images
                    try:
                        plt.subplot(4, 8, i + 25)
                        plt.imshow(self.base_teste.x_test[correct_indices[::-1][i]].reshape(200, 200), cmap='gray')
                        plt.title(
                            f"True {i}: {self.base_teste.y_test[correct_indices[::-1][i]]}, Pred:{labels_resultado[correct_indices[::-1][i]]}")
                    except:
                        pass

                plt.tight_layout()
                plt.show()

                # Plot using seaborn
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Kidney_stone'],
                            yticklabels=['Normal', 'Kidney_stone'])
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title(f"Matriz Confusão do Classificador {str(autoencoder.id).zfill(3)}")

                nm_diretorio = f'RESULTADOS/{NOME_PROCESSO}/CLASSIFICADOR/{str(autoencoder.id).zfill(3)}'
                os.makedirs(f'{nm_diretorio}', exist_ok=True)
                plt.savefig(f'{nm_diretorio}/matriz.png')
                with open(f"{nm_diretorio}/resultado.txt", "w") as file:
                    file.write(str(accuracy_score(self.base_teste.y_test, labels_resultado)) + '\n')

                    file.write(f"corretos: {str(correct_indices)}\n")
                    file.write(f"incorretos: {str(missed_indices)}")


        else:
            classificador = self.get_classificador()
            self.resultado = []

            base_teste = tf.reshape(self.base_teste.x_test, (-1,) + self.base_teste.x_test[0].shape)
            rf_classifier = classificador

            for autoencoder in self.pool.pool:
                x_train_flat = np.array(self.base_treino.x_train)
                x_treino_encoded = autoencoder.encoder.predict(x_train_flat, batch_size=len(self.base_treino.x_train))
                rf_classifier.fit(x_treino_encoded, self.base_treino.y_train)
                resultado = autoencoder.encoder.predict(base_teste)
                predicoes = rf_classifier.predict_proba(resultado)
                self.resultado.append(predicoes)

                labels_resultado = np.argmax(predicoes, axis=1)
                cm = confusion_matrix(self.base_teste.y_test, labels_resultado)

                # Plot using seaborn
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Kidney_stone'], yticklabels=['Normal', 'Kidney_stone'])
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title(f"Matriz Confusão do Classificador {str(autoencoder.id).zfill(3)}")

                nm_diretorio = f'RESULTADOS/{NOME_PROCESSO}/CLASSIFICADOR/{str(autoencoder.id).zfill(3)}'
                os.makedirs(f'{nm_diretorio}', exist_ok=True)
                plt.savefig(f'{nm_diretorio}/matriz.png')
                with open(f"{nm_diretorio}/resultado.txt", "w") as file:
                    file.write(str(accuracy_score(self.base_teste.y_test, labels_resultado)))

            self.calcular_acuraria()

            return True

    def calcular_acuraria(self):
        prod = np.product(self.resultado, axis=0)
        lista_produto = [np.argmax(x) for x in prod]

        sum = np.sum(self.resultado, axis=0)
        lista_soma = [np.argmax(x) for x in sum]

        nm_diretorio = f'RESULTADOS/{NOME_PROCESSO}/POOL'
        with open(f"{nm_diretorio}/resultado.txt", "w") as file:
            file.write(f"SOMA: {str(accuracy_score(self.base_teste.y_test, lista_soma))}\n")
            file.write(f"PRODUTO: {str(accuracy_score(self.base_teste.y_test, lista_produto))}\n")