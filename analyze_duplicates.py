"""Analisis metodologico: imagenes duplicadas entre train y test.

El dataset oficial contiene imagenes cuyo contenido binario es identico en
train y test (mismas fotos copiadas en los dos splits). Esto contamina las
metricas: un KNN puede "acertar" perfectamente esas imagenes porque las ha
memorizado, no porque haya generalizado.

Este script:

1. Recorre los archivos referenciados por images/gt.json en train y test.
2. Calcula el hash SHA-256 del contenido binario de cada .jpg.
3. Encuentra los hashes presentes en ambos splits → imagenes duplicadas.
4. Recalcula el accuracy del KNN en el test SIN esas duplicadas, para tener
   una metrica "limpia" comparable con el accuracy oficial.

NO modifica el dataset ni el split oficial. Solo reporta. Los resultados
se guardan en informe/duplicates_analysis.json para citarlos en el informe.
"""

import hashlib
import json
import os
import time

import numpy as np

from KNN import KNN
from utils_data import read_dataset


ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(ROOT, 'images')
GT_PATH = os.path.join(IMAGES_DIR, 'gt.json')
OUT_PATH = os.path.join(ROOT, 'informe', 'duplicates_analysis.json')


def hash_file(path):
    """SHA-256 del contenido binario de un archivo. Bloque de 64KB es suficiente
    para los .jpg pequenos del dataset (cada uno ~3-5KB)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def hashes_por_split(gt, split):
    """Devuelve {nombre_imagen: hash} para un split del ground truth."""
    out = {}
    for nombre in gt[split].keys():
        path = os.path.join(IMAGES_DIR, split, nombre + '.jpg')
        if os.path.exists(path):
            out[nombre] = hash_file(path)
    return out


def encontrar_duplicados(hashes_train, hashes_test):
    """Devuelve la lista de (test_name, train_name) que comparten hash.

    Un nombre de test puede coincidir con varios de train (en teoria); cogemos
    el primero por orden alfabetico para tener un mapping reproducible.
    """
    train_por_hash = {}
    for nombre, h in hashes_train.items():
        train_por_hash.setdefault(h, []).append(nombre)

    duplicados = []
    for nombre_test, h in hashes_test.items():
        if h in train_por_hash:
            duplicados.append((nombre_test, sorted(train_por_hash[h])[0]))
    return duplicados


def accuracy(pred, real):
    return 100.0 * np.sum(pred == real) / len(real)


def main():
    print("=== Analisis de duplicados train/test ===\n")
    print("1) Calculando hashes SHA-256 de los archivos del gt.json...")
    with open(GT_PATH) as f:
        gt = json.load(f)

    t0 = time.time()
    hashes_train = hashes_por_split(gt, 'train')
    hashes_test = hashes_por_split(gt, 'test')
    print(f"   train: {len(hashes_train)} archivos hasheados")
    print(f"   test:  {len(hashes_test)} archivos hasheados")
    print(f"   tiempo: {time.time() - t0:.1f}s\n")

    print("2) Cruzando hashes...")
    duplicados = encontrar_duplicados(hashes_train, hashes_test)
    n_dup = len(duplicados)
    pct = 100.0 * n_dup / len(hashes_test)
    print(f"   imagenes en test duplicadas en train: {n_dup}")
    print(f"   porcentaje sobre el test: {pct:.1f}%\n")

    nombres_test_duplicados = {nt for (nt, _) in duplicados}

    print("3) Recalculando accuracy del KNN sin los duplicados...")
    train_imgs, train_class, _, test_imgs, test_class, _ = read_dataset(
        root_folder=IMAGES_DIR + '/', gt_json=GT_PATH,
    )
    # read_dataset baraja con seed=42 y NO devuelve los nombres; reconstruimos
    # la mascara cargando los nombres en el mismo orden que el JSON antes del
    # shuffle, y aplicando la misma permutacion.
    test_names_ordered = list(gt['test'].keys())
    np.random.seed(42)
    # Al leer el dataset, primero se cargan las imagenes en el orden del JSON
    # y despues se baraja con seed=42 (ver read_dataset). Replicamos solo el
    # shuffle del test (el de train usa una segunda llamada de random.shuffle
    # asi que avanzamos la seed primero).
    train_perm = np.arange(len(gt['train']))
    np.random.shuffle(train_perm)
    test_perm = np.arange(len(gt['test']))
    np.random.shuffle(test_perm)
    test_names_shuffled = [test_names_ordered[i] for i in test_perm]

    es_duplicada = np.array(
        [nombre in nombres_test_duplicados for nombre in test_names_shuffled]
    )
    n_limpio = int((~es_duplicada).sum())
    n_dup_en_split = int(es_duplicada.sum())
    print(f"   test total: {len(test_class)} | duplicadas en orden barajado: {n_dup_en_split}"
          f" | unicas: {n_limpio}")

    # KNN con k=3 (el recomendado en el informe) sobre todo el test
    knn = KNN(train_imgs, train_class)
    pred_full = knn.predict(test_imgs, k=3)
    acc_oficial = accuracy(pred_full, test_class)

    # Accuracy solo sobre las unicas (mascara booleana)
    pred_limpio = pred_full[~es_duplicada]
    real_limpio = test_class[~es_duplicada]
    acc_limpio = accuracy(pred_limpio, real_limpio)

    # Accuracy solo sobre los duplicados (espera ~100% si el KNN los memoriza)
    pred_dup = pred_full[es_duplicada]
    real_dup = test_class[es_duplicada]
    acc_dup = accuracy(pred_dup, real_dup) if len(pred_dup) > 0 else None

    print(f"\n   Accuracy oficial (test entero, k=3):   {acc_oficial:.2f}%")
    print(f"   Accuracy sin duplicados (k=3):         {acc_limpio:.2f}%")
    if acc_dup is not None:
        print(f"   Accuracy solo en duplicadas (k=3):     {acc_dup:.2f}%")

    print("\n4) Guardando resultado en informe/duplicates_analysis.json...")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    resumen = {
        'n_train_archivos': len(hashes_train),
        'n_test_archivos': len(hashes_test),
        'n_duplicados': n_dup,
        'porcentaje_test_duplicado': round(pct, 2),
        'accuracy_oficial_k3': round(acc_oficial, 2),
        'accuracy_sin_duplicados_k3': round(acc_limpio, 2),
        'accuracy_solo_duplicados_k3': round(acc_dup, 2) if acc_dup is not None else None,
        'ejemplos_duplicados': duplicados[:10],
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(resumen, f, indent=2)
    print(f"   listo: {OUT_PATH}")


if __name__ == '__main__':
    main()
