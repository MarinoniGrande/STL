import datetime
import random
import json
import sys
import os
import datetime
import shutil

ARQUIVO_CONFIGURACOES = 'padrao'
NOME_PROCESSO = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def criar_processo(tipo=None):
    os.makedirs(f'RESULTADOS/{NOME_PROCESSO}/', exist_ok=True)
    os.makedirs(f'RESULTADOS/{NOME_PROCESSO}/POOL/', exist_ok=True)
    os.makedirs(f'RESULTADOS/{NOME_PROCESSO}/CLASSIFICADOR/', exist_ok=True)
    os.makedirs(f'RESULTADOS/{NOME_PROCESSO}/RECONSTRUCAO/', exist_ok=True)
    shutil.copy(f'CONFIGURACOES/{ARQUIVO_CONFIGURACOES}.json', f'RESULTADOS/{NOME_PROCESSO}/configuracoes.json')
    with open(f'RESULTADOS/{NOME_PROCESSO}/tipo.txt', 'w') as f:
        f.write(tipo)


def get_padrao(variavel=None):
    """
    Função que recebe uma variavel e retorna seu valor padrào do projeto
    :param variavel: Variável que deseja retornar o valor padrão
    :return: Valor padrão da variável
    """
    nome_arquivo = f'CONFIGURACOES/{ARQUIVO_CONFIGURACOES}.json'
    try:

        with open(nome_arquivo) as f:
            data = json.load(f)
    except:
        sys.exit(F"ERRO -1: ERRO AO ENCONTRAR ARQUIVO '{f'CONFIGURACOES/{ARQUIVO_CONFIGURACOES}.json'}'")

    try:
        variaveis = variavel.split('.')
        valor = data
        for v in variaveis:
            valor = valor[v]

        return valor

    except:
        sys.exit(F"ERRO -2: ERRO AO ENCONTRAR VARIÁVEL '{variavel}'")


def get_valor_aleatorio(lista=None):
    """
    Função que recebe lista e retorna um valor aleatório dessa lista
    :param lista: Lista com dados aleatórios
    :return: Valor aleatório da lista
    """
    if lista is None:
        return None

    numero_aleatorio = random.randint(0, len(lista) - 1)

    return lista[numero_aleatorio]



def configurar_reprodutibilidade():
    """
    Função que seta todas as variáveis de reprodutibilidade, se a flag estiver ativa nas configurações
    :return: status da configuração
    """
    print('Configurando reprodutibilidade')
    if not get_padrao('SEEDS.IS_ALEATORIO'):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
        os.environ['TF_DETERMINISTIC_OPS'] = 'true'

        import numpy as np
        import tensorflow as tf
        import cv2
        import keras

        np.random.seed(get_padrao('SEEDS.NUMPY'))
        random.seed(get_padrao('SEEDS.NUMPY'))
        tf.random.set_seed(get_padrao('SEEDS.TENSORFLOW'))
        cv2.setRNGSeed(get_padrao('SEEDS.OPENCV'))
        keras.utils.set_random_seed(get_padrao('SEEDS.KERAS'))

    return True