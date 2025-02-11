from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

from BO.util.util import get_padrao


class Classificador:
    def __init__(self, base=None, pool=None, classificador=None, estimator=None):
        """
        Classe responsável de fazer a classificação das imagens alvo
        :param base: Base de dado alvo
        :param classificador: Classificador utilizado
        :param estimator: Estimator usado no classificador
        """
        self.base = base
        self.x = []
        self.classificador = classificador if classificador is not None else get_padrao('CLASSIFICADOR.CLASSIFICADOR')
        self.estimator = estimator if estimator is not None else get_padrao('CLASSIFICADOR.ESTIMATOR')
        self.resultado = []
        self.acuracia = 0.0

        self.pool = pool
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

    def get_classificador(self):
        dict_classificadores = {
            'bagging': BaggingClassifier(estimator=self.get_estimator(), n_estimators=get_padrao('CLASSIFICADOR.N_ESTIMATORS'), random_state=get_padrao('CLASSIFICADORRANDOM_STATE')),
            'randomforest': RandomForestClassifier(n_estimators=get_padrao('CLASSIFICADOR.N_ESTIMATORS'), random_state=get_padrao('SEEDS.CLASSIFICADOR')),
            'svm': SVC(probability=True, random_state=get_padrao('SEEDS.CLASSIFICADOR'))
        }
        return dict_classificadores.get(self.classificador)

    def get_estimator(self):
        dict_estimators = {
            'decisiontree': DecisionTreeClassifier(random_state=get_padrao('SEEDS.CLASSIFICADOR'))
        }
        return dict_estimators.get(self.estimator)

    def dividir_base(self):
        self.carregar_x()
        self.x = np.concatenate(self.x, axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.base.y_train, test_size=0.3, random_state=42)

    def carregar_x(self):
        self.x = []
        for x in self.pool:
            if get_padrao('DEBUG'):
                print(f'Carregando encoder {x.id}')
            self.x.append(np.load(f"AUTOENCODERS/{get_padrao('BASE.DIRETORIO_TREINO')}/{get_padrao('POOL.MODELAGEM')}/encoder_{str(x.id).zfill(3)}.npy"))
        return self.x

    def classificar(self):
        classificador = self.get_classificador()

        self.dividir_base()

        classificador.fit(self.x_train, self.y_train)
        self.resultado = classificador.predict(self.x_test)

        self.calcular_acuraria()

        if get_padrao('DEBUG'):
            self.printar_matrix_confusao()
            print(f'Acurácia: {self.acuracia}')

    def printar_matrix_confusao(self):
        cm = confusion_matrix(self.y_test, self.resultado)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Normal', 'Kidney_stone'], yticklabels=['Normal', 'Kidney_stone'])
        plt.title(f"Matrix de Confusão do classificador {self.classificador}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def calcular_acuraria(self):
        self.acuracia = accuracy_score(self.y_test, self.resultado)
