from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

from BO.util.util import get_padrao, NOME_PROCESSO
from BO.base.base import Base


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

        self.base_teste = Base(is_normalizar=True, tipo='labeled', is_base_separada=get_padrao('BASE.IS_DIRETORIO_ALVO_TESTE_SEPARADO'), diretorio=f"BASE/{get_padrao('BASE.DIRETORIO_ALVO_TESTE')}")
        self.base_treino = Base(is_normalizar=True, tipo='labeled', is_base_separada=get_padrao('BASE.IS_DIRETORIO_ALVO_TREINO_SEPARADO'), diretorio=f"BASE/{get_padrao('BASE.DIRETORIO_ALVO_TREINO')}")
        self.pool = pool

        self.carregar_bases()

    def carregar_bases(self):
        self.base_treino.carregar()
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

    def classificar(self):
        classificador = self.get_classificador()
        self.resultado = []

        base_teste = tf.reshape(self.base_teste.x_test, (-1,) + self.base_teste.x_test[0].shape)
        rf_classifier = classificador

        for autoencoder in self.pool:
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