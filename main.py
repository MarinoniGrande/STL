import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import BO.util.util
from BO.base.base import Base
from BO.pool.pool import Pool
from BO.classificador.classificador import Classificador


BO.util.util.ARQUIVO_CONFIGURACOES, tipo = sys.argv[1], sys.argv[2]

BO.util.util.configurar_reprodutibilidade()

BO.util.util.criar_processo(tipo=tipo)

base = Base(is_normalizar=True, tipo='unlabeled', diretorio=f"BASE/{BO.util.util.get_padrao('BASE.DIRETORIO_TREINO')}")
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
        print('teste')
        import albumentations as A  # Albumentations for image augmentation
        import cv2  # OpenCV for image processing
        import numpy as np  # NumPy for array manipulations
        import matplotlib.pyplot as plt
        # Transformações escolhidas:
        transform = A.Compose([
            A.RandomRain(
                drop_length=8, drop_width=1,
                drop_color=(180, 180, 180), blur_value=5, brightness_coefficient=0.8, p=0.15
            ),
            A.GaussNoise(p=0.15),
            A.ChannelShuffle(p=0.15),
            A.Rotate(limit=40, p=0.15),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.15),
            A.AdvancedBlur(blur_limit=(7, 9), noise_limit=(0.75, 1.25), p=0.15),
            # A.Resize(height=64, width=64)
        ])
        def albumentations(img):
            """
            Faz a transformação da imagem a partir do transform definido
            """
            data = {"image": img}
            augmented = transform(**data)  # * para expandir o dicionário
            return augmented['image']

        for image in base.x_train:
            a = albumentations(img=image)
            plt.figure(figsize=(20, 16))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.subplot(1, 2, 2)
            plt.imshow(a)
            plt.axis("off")  # Oculta os eixos
            plt.show()

            print(a)

    else:
        sys.exit(F"ERRO -3: TIPO NÃO DEFINIDO NO CÓDIGO: '{tipo}'")
except Exception as e:
    with open(f'RESULTADOS/{BO.util.util.NOME_PROCESSO}/erro.txt', 'w') as f:
        f.write(str(e))