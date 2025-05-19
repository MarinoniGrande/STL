import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/nonexistent"

import BO.util.util
from BO.base.base import Base
from BO.pool.pool import Pool
from BO.classificador.classificador import Classificador



import tensorflow as tf



BO.util.util.ARQUIVO_CONFIGURACOES, tipo = sys.argv[1], sys.argv[2]


gpus = tf.config.list_physical_devices('GPU')
print('gpus')
print(gpus)
print(f'Encontrado {len(gpus)} GPU{"S" if len(gpus) > 1 else ""}')
usado = 0
if gpus:
    try:
        print(gpus)
        if BO.util.util.get_padrao('GPU') is None:
            for gpu in gpus:
                print(gpu)
                usado += 1
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            usado  = len(gpus)
            tf.config.set_visible_devices(gpus[BO.util.util.get_padrao('GPU')], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[BO.util.util.get_padrao('GPU')], True)

    except RuntimeError as e:
        print(e)

print(f'Utilizado {usado} GPU{"S" if len(gpus) > 1 else ""}')

BO.util.util.configurar_reprodutibilidade()

BO.util.util.criar_processo(tipo=tipo)

base = Base(is_normalizar=True, tipo='unlabeled', diretorio=f"{BO.util.util.get_padrao('BASE.DIRETORIO_TREINO')}", is_augmentation=True)
_ , _ = base.carregar()

try:
    if tipo == 'criar':
        _ = Pool(base=base).criar()

    elif tipo == 'classificar':
        pool = Pool(base=base)
        _ = pool.carregar_pool(tipo='encoder')
        #_ = pool.aplicar_funcao_custo_offline()
        # salvar best weights
        _ = Classificador(pool=pool).classificar()

    elif tipo == 'reconstruir':
        #base = Base(is_normalizar=True, tipo='labeled', is_base_separada=BO.util.util.get_padrao('BASE.IS_DIRETORIO_ALVO_TREINO_SEPARADO'), diretorio=f"BASE/{BO.util.util.get_padrao('BASE.DIRETORIO_ALVO_TREINO')}")
        #_, _ = base.carregar()
        pool = Pool(base=base)
        #_ = pool.visualizar_reconstrucao(15)
        _ = pool.carregar_pool(tipo='autoencoder')
        _ = pool.carregar_imagens_reconstrucao(qtd_imagens_reconstrucao=15)
        for aec in pool.pool:
            _ = pool.aplicar_funcao_custo_online(autoencoder=aec)
    elif tipo == 'outro':
        import tensorflow as tf

        # List all physical devices TensorFlow sees
        print("Available devices:", tf.config.list_physical_devices())
        # Check for GPUs
        print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

    else:
        sys.exit(F"ERRO -3: TIPO NÃO DEFINIDO NO CÓDIGO: '{tipo}'")
except Exception as e:
    with open(f'RESULTADOS/{BO.util.util.NOME_PROCESSO}/erro.txt', 'w') as f:
        f.write(str(e))