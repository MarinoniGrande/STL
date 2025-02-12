import os
import sys
import BO.util.util
from BO.base.base import Base
from BO.pool.pool import Pool
from BO.classificador.classificador import Classificador

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
BO.util.util.ARQUIVO_CONFIGURACOES, tipo = sys.argv[1], sys.argv[2]

BO.util.util.configurar_reprodutibilidade()

base = Base(is_normalizar=True, tipo='unlabeled', diretorio=f"BASE/{BO.util.util.get_padrao('BASE.DIRETORIO_TREINO')}")
_ , _ = base.carregar()

if tipo == 'criar':
    _ = Pool(base=base).criar()

elif tipo == 'classificar':
    BO.util.util.criar_processo()

    pool = Pool(base=base)
    _ = pool.carregar_pool(tipo='encoder')
    _ = pool.aplicar_funcao_custo_offline()

    _ = Classificador(pool=pool).classificar()

else:
    sys.exit(F"ERRO -3: TIPO NÃO DEFINIDO NO CÓDIGO: '{tipo}'")