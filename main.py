import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/nonexistent"

import BO.util.util
from BO.base.base import Base
from BO.pool.pool import Pool
import tensorflow as tf
from BO.metrics.metrics import PerfTimer, build_report, save_metrics_report

BO.util.util.ARQUIVO_CONFIGURACOES, tipo = sys.argv[1], sys.argv[2]

if BO.util.util.get_padrao('NOME_PROCESSO') is not None:
    BO.util.util.NOME_PROCESSO = BO.util.util.get_padrao('NOME_PROCESSO')


from BO.classificador.classificador import Classificador

gpus = tf.config.list_physical_devices('GPU')
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
            usado  = 1
            tf.config.set_visible_devices(gpus[BO.util.util.get_padrao('GPU')], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[BO.util.util.get_padrao('GPU')], True)

    except RuntimeError as e:
        print(e)

# === start metrics timer ===
_timer = PerfTimer().__enter__()   # manual enter so we can close in finally

_error_trace = None


print(f'Utilizado {usado} GPU{"S" if len(gpus) > 1 else ""}')

BO.util.util.configurar_reprodutibilidade()

BO.util.util.criar_processo(tipo=tipo)

# base = Base(is_normalizar=True, tipo='unlabeled',
#                     diretorio=f"{BO.util.util.get_padrao('BASE.DIRETORIO_TREINO')}", is_augmentation=False, is_base_separada=BO.util.util.get_padrao('BASE.IS_DIRETORIO_TREINO_SEPARADO'))
# _ , _ = base.carregar()

try:
    if tipo == 'criar':
        _ = Pool(base=base).criar()

    elif tipo == 'arquivo':
        from pathlib import Path
        import shutil
        from sklearn.model_selection import train_test_split

        SRC = Path("/home/aghochuli/ngrande/data/base/LITIASE/YILDRIM")
        DST = Path("/home/aghochuli/ngrande/data/base/LITIASE/YILDRIM_NICOLAS")
        CLASSES = ["Kidney_stone", "Normal"]
        TEST_SIZE = 0.30
        SEED = 42
        MODE = "copy"  # copy | move

        EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


        def collect_images(class_name: str):
            imgs = []
            for split in ["Train", "Test"]:
                d = SRC / split / class_name
                if d.exists():
                    imgs.extend([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXTS])
            return imgs


        def ensure_dirs():
            for split in ["Train", "Test"]:
                for cls in CLASSES:
                    (DST / split / cls).mkdir(parents=True, exist_ok=True)


        def place(src_path: Path, out_dir: Path):
            out_path = out_dir / src_path.name
            if out_path.exists():
                stem, suf = out_path.stem, out_path.suffix
                k = 1
                while out_path.exists():
                    out_path = out_path.with_name(f"{stem}__dup{k}{suf}")
                    k += 1
            if MODE == "copy":
                shutil.copy2(src_path, out_path)
            else:  # move
                shutil.move(str(src_path), str(out_path))


        ensure_dirs()
        for cls in CLASSES:
            imgs = collect_images(cls)
            if not imgs:
                print(f"No images found for {cls}")
                continue

            # sklearn split
            train_files, test_files = train_test_split(
                imgs, test_size=TEST_SIZE, random_state=11
            )

            for f in train_files:
                place(f, DST / "Train" / cls)
            for f in test_files:
                place(f, DST / "Test" / cls)

            print(f"{cls}: {len(train_files)} train, {len(test_files)} test")


    elif tipo == 'classificar':
        print('iniciando classificacao...')
        pool = Pool(base=base, diretorio=BO.util.util.get_padrao('POOL.DIRETORIO'))
        _ = pool.carregar_pool(tipo='encoder')
        #_ = pool.aplicar_funcao_custo_offline()
        # salvar best weights

        base_test = Base(is_normalizar=True, tipo='unlabeled',
                    diretorio=f"{BO.util.util.get_padrao('BASE.DIRETORIO_TREINO')}", is_augmentation=False, is_base_separada=BO.util.util.get_padrao('BASE.IS_DIRETORIO_TREINO_SEPARADO'))
        _, _ = base_test.carregar(is_split_validacao=True)

        _ = Classificador(pool=pool).classificar(base_test=base_test)

    elif tipo == 'reconstruir':
        #base = Base(is_normalizar=True, tipo='labeled', is_base_separada=BO.util.util.get_padrao('BASE.IS_DIRETORIO_ALVO_TREINO_SEPARADO'), diretorio=f"BASE/{BO.util.util.get_padrao('BASE.DIRETORIO_ALVO_TREINO')}")
        #_, _ = base.carregar()
        pool = Pool(base=base, diretorio=BO.util.util.get_padrao('POOL.DIRETORIO'))
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

# finish timer & write metrics regardless of success/failure
PerfTimer.__exit__(_timer, None, None, None)

report = build_report(
    _timer,
    notes="Métricas pós-execução (CPU, RAM, disco, rede, GPU, pacotes).",
    error=_error_trace,
)
path = save_metrics_report(os.path.join("RESULTADOS", BO.util.util.NOME_PROCESSO), report, filename="METRICAS.json")
print(f"[METRICS] Report saved to: {path}")