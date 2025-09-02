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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

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

        threshold_similaridade = get_padrao('POOL.VALOR_CUSTO_THRESHOLD_OFFLINE')
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

        accs_por_encoder = []
        accs_por_encoder_filtrados = []
        encoder_ids = []
        encoder_ids_filtrados = []

        for p in novo_pool:
            print(p.id)
            encoder = p.encoder
            if self.classificador == 'PIPELINE':
                x_train_encoded = encoder.predict(x_train_flat, batch_size=16)

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=0.95, svd_solver="full")),
                    ("svc", SVC(kernel="rbf", C=10, gamma=0.01, class_weight="balanced",
                                probability=False, cache_size=1000, random_state=42))
                ])
                pipe.fit(x_train_encoded, base_treino_cla.y_train)

                y_pred_enc = encoder.predict(x_test)
                cal = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
                cal.fit(x_train_encoded, base_treino_cla.y_train)
                predicoes = cal.predict_proba(y_pred_enc)

            elif self.classificador == 'PIPELINE+':
                x_train_encoded = encoder.predict(x_train_flat, batch_size=16)
                X_tr = x_train_encoded  # latent vectors (n_samples, n_features)
                y_tr = base_treino_cla.y_train
                X_te = encoder.predict(x_test)  # latents for test set

                # 1) Define a strong base pipeline (normalize -> (optional PCA) -> SVC)
                base_pipe = Pipeline([
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("norm", Normalizer(norm="l2")),  # helps on latents
                    ("pca", PCA(n_components=0.95, whiten=True)),  # will be tuned (possibly disabled)
                    ("svc", SVC(kernel="rbf", probability=False, class_weight="balanced",
                                cache_size=1000, random_state=42))
                ])

                # 2) Search space — include option to skip PCA by passing None via "passthrough"
                param_grid = [
                    {
                        "pca": [PCA(n_components=0.95, whiten=True),
                                PCA(n_components=0.99, whiten=True),
                                "passthrough"],
                        "svc__C": [0.5, 1, 3, 10, 30, 100],
                        "svc__gamma": ["scale", "auto", 1e-3, 3e-3, 1e-2]
                    }
                ]
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                grid = GridSearchCV(
                    estimator=base_pipe,
                    param_grid=param_grid,
                    scoring="roc_auc",  # optimize for ranking quality
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True,
                    verbose=0
                )
                grid.fit(X_tr, y_tr)

                #best_pipe = grid.best_estimator_

                # 3) Calibrate probabilities with CV (no peeking at test)
                # Use sigmoid if sample size is modest; isotonic if you have lots of data.
                cal = CalibratedClassifierCV(
                    #base_estimator=best_pipe,
                    method="sigmoid",  # try "isotonic" if you have enough data
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                )
                cal.fit(X_tr, y_tr)

                # 4) Predict calibrated probabilities on test latents
                predicoes = cal.predict_proba(X_te)[:, 1]

            elif self.classificador == 'SVC':
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

            # === ADDED: track IDs and per-encoder accuracy ===
            encoder_ids.append(p.id)
            if p.id in encoders_filtrados:
                encoder_ids_filtrados.append(p.id)

            acc_i = accuracy_score(base_teste_cla.y_test, y_pred)
            accs_por_encoder.append((p.id, float(acc_i)))
            if p.id in encoders_filtrados:
                accs_por_encoder_filtrados.append((p.id, float(acc_i)))

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

            # === ADDED: per-encoder accuracies ===
            f.write("\n=== ACURÁCIA POR ENCODER (TODOS) ===\n")
            for enc_id, acc in sorted(accs_por_encoder, key=lambda x: x[1], reverse=True):
                f.write(f"encoder_id={enc_id} | acc={acc:.6f}\n")

            f.write("\n=== ACURÁCIA POR ENCODER (FILTRADOS) ===\n")
            if accs_por_encoder_filtrados:
                for enc_id, acc in sorted(accs_por_encoder_filtrados, key=lambda x: x[1], reverse=True):
                    f.write(f"encoder_id={enc_id} | acc={acc:.6f}\n")
            else:
                f.write("(vazio)\n")

            # === ADDED: lists of encoder IDs ===
            f.write("\n=== LISTA DE ENCODERS (TODOS) ===\n")
            f.write(", ".join(map(str, encoder_ids)) + "\n")

            f.write("\n=== LISTA DE ENCODERS (FILTRADOS) ===\n")
            f.write((", ".join(map(str, encoder_ids_filtrados)) + "\n") if encoder_ids_filtrados else "(vazio)\n")

            # === ADDED: statistics over per-encoder accuracies ===
            stats_total = self._resumo_stats([acc for _, acc in accs_por_encoder])
            stats_filtrado = self._resumo_stats([acc for _, acc in accs_por_encoder_filtrados])

            f.write("\n=== ESTATÍSTICAS (TODOS) ===\n")
            if stats_total:
                for k, v in stats_total.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write("sem dados\n")

            f.write("\n=== ESTATÍSTICAS (FILTRADOS) ===\n")
            if stats_filtrado:
                for k, v in stats_filtrado.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write("sem dados\n")

        print('7')
        return True

    def _resumo_stats(self,vals):
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return None
        media = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        mediana = float(np.median(arr))
        mad = float(np.median(np.abs(arr - mediana)))  # Median Absolute Deviation
        mean_abs_dev = float(np.mean(np.abs(arr - media)))  # Desvio Médio Absoluto
        minimo = float(np.min(arr))
        maximo = float(np.max(arr))
        return {
            "count": int(arr.size),
            "mean": media,
            "std": std,
            "median": mediana,
            "MAD": mad,
            "mean_abs_dev": mean_abs_dev,
            "min": minimo,
            "max": maximo,
        }

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