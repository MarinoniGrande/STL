import os
import sys
import BO.util.util
from BO.base.base import Base
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
BO.util.util.ARQUIVO_CONFIGURACOES = sys.argv[1]

BO.util.util.configurar_reprodutibilidade()


base = Base(is_normalizar=True, tipo='unlabeled', diretorio=f"BASE/{BO.util.util.get_padrao('BASE.DIRETORIO_TREINO')}")
_ , _ = base.carregar()
#
# base_treino_classificador = Base(is_normalizar=True, tipo='labeled', is_base_separada=BO.util.util.get_padrao('BASE.IS_DIRETORIO_ALVO_TREINO_SEPARADO'), diretorio=f"BASE/{BO.util.util.get_padrao('DIRETORIO_ALVO_TESTE')}")
# _ , _ = base_treino_classificador.carregar()
#
# base_teste_classificador = Base(is_normalizar=True, tipo='labeled', is_base_separada=BO.util.util.get_padrao('BASE.IS_DIRETORIO_ALVO_TESTE_SEPARADO'), diretorio=f"BASE/{BO.util.util.get_padrao('BASE_TESTE_DIRETORIO')}")
# _ , _ = base_teste_classificador.carregar()
# base_teste = tf.reshape(base_teste_classificador.x_test, (-1,) + base_teste_classificador.x_test[0].shape)


# Criar um pool novo, rodar só se quiser criar algo
from BO.pool.pool import Pool
pool = Pool(base=base)
_ = pool.criar()


# Carregar um pool ja existente a partir de uma base
# pool = Pool(base=base)
# _ = pool.carregar_pool(tipo='encoder')
# pool.aplicar_funcao_custo_offline()

# # FINETUNING
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#
# for d in [
#     #"/content/drive/MyDrive/Mestrado/autoencoders/kyoto/s",
#     #"/content/drive/MyDrive/Mestrado/autoencoders/kyoto/l",
#     #"/content/drive/MyDrive/Mestrado/autoencoders/kyoto/a",
#     #"/content/drive/MyDrive/Mestrado/autoencoders/kyoto/sl",
#     # "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/la",
#     # "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/sa",
#     "/content/drive/MyDrive/Mestrado/AEC/KYOTO/SLA",
# ]:
#   print(f'POOL: {d}')
#   pool = None
#   pool = Pool(base=base, diretorio=d)
#   _ = pool.carregar_pool(tipo='encoder')
#   x_fine = tf.reshape(base_labeled.x_train, (-1,) + base_labeled.x_train[0].shape)
#   x_val = tf.reshape(base_labeled.x_val, (-1,) + base_labeled.x_val[0].shape)
#
#   for aec in pool.pool:
#     print(f"ENCODER: {aec.id}")
#     if 1==1:
#       latent_fine = aec.encoder.predict(x_fine)
#       latent_val = aec.encoder.predict(x_val)
#       aec.encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
#       aec.encoder.fit(x_fine, latent_fine, validation_data=(x_val, latent_val), epochs=150, batch_size=32, shuffle=True, callbacks=[early_stopping])
#       nm_arquivo = f"{d}/FINETUNNING/{get_padrao('BASE_ALVO')}/encoder_{str(aec.id).zfill(3)}"
#       print(nm_arquivo)
#       aec.encoder.save_weights(f"{nm_arquivo}.weights.h5")



# for d in [
#    # "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/s",
#     # "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/l",
#     # "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/a",
#     #"/content/drive/MyDrive/Mestrado/autoencoders/kyoto/sl",
#     # "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/la",
#     # "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/sa",
#     "/content/drive/MyDrive/Mestrado/AEC/KYOTO/SLA"
# ]:
#   print(d)
#   pool = Pool(base=base, diretorio=d)
#   _ = pool.carregar_pool(tipo='encoder', is_finetunning=True)
#   #pool.aplicar_funcao_custo_offline()
#   lista_resultados = avaliar_autoencoders(pool=pool, base_treino=base_treino_classificador, base_teste=base_teste)
#
#   cc, ee = 0,0
#   soma = np.sum([lista_resultados], axis=1)
#   soma = soma.reshape(soma.shape[1], soma.shape[2])
#   resultado_previsto = np.argmax(soma, axis=1)
#   for m in range(0, len(resultado_previsto)):
#       b = resultado_previsto[m]
#       c = base_teste_classificador.y_test[m]
#       if (b == c):
#           cc = cc + 1
#       else:
#           ee = ee + 1
#   res_soma = cc / (cc + ee)
#   print(f'Soma: {res_soma}')
#
#   cc, ee = 0,0
#   prod = np.product([lista_resultados], axis=1)
#   prod = prod.reshape(prod.shape[1], prod.shape[2])
#   resultado_previsto = np.argmax(prod, axis=1)
#   for m in range(0, len(resultado_previsto)):
#       b = resultado_previsto[m]
#       c = base_teste_classificador.y_test[m]
#       if (b == c):
#           cc = cc + 1
#       else:
#           ee = ee + 1
#   res_prod = cc / (cc + ee)
#   print(f'Prod: {res_prod}')


# def avaliar_autoencoders(pool=None, base_treino=None, base_teste=None):
#     lista_predicoes = []
#     for autoencoder in pool.pool:
#         rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#         x_train_flat = np.array(base_treino.x_train)
#         x_treino_encoded = autoencoder.encoder.predict(x_train_flat, batch_size=len(base_treino.x_train))
#         rf_classifier.fit(x_treino_encoded, base_treino.y_train)
#         resultado = autoencoder.encoder.predict(base_teste)
#         predicoes = rf_classifier.predict_proba(resultado)
#         lista_predicoes.append(predicoes)
#
#     return lista_predicoes