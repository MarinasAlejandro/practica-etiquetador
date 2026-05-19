"""Cuatro analisis de eficiencia para el informe.

1. Inicializacion de centroides en KMeans (first vs random):
   numero de iteraciones hasta convergencia y WCD final.
2. Valor de K en KNN: accuracy en el conjunto de test para k = 1..11.
3. Criterio de find_bestK: comparar el umbral del 20% del PDF con
   otros umbrales (10%, 15%, 25%, 30%) y ver si cambia la K elegida.
4. Normalizacion Min-Max en el KNN: comparar el accuracy con y sin
   normalizar los valores de los pixeles, siguiendo la teoria del
   Bloc 4 sesion 7 sobre la importancia de la escala en algoritmos
   basados en distancias.

Los resultados se imprimen por consola y se guardan en
informe/results.json para incluirlos en el informe.
"""

import json
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from KNN import KNN
import Kmeans
import utils
from utils_data import read_dataset

IMAGES_DIR = os.path.join(ROOT, 'images')
GT_PATH = os.path.join(IMAGES_DIR, 'gt.json')
RESULTS_PATH = os.path.join(ROOT, 'informe', 'results.json')


# ====================================================================
# Helpers
# ====================================================================

def medir_accuracy(pred, real):
    return 100.0 * np.sum(pred == real) / len(real)


# ====================================================================
# Analisis 1 — Inicializacion de centroides en KMeans
# ====================================================================

def analisis_1_inicializacion(imagenes, n_imagenes=50, K=4):
    """Compara 'first' vs 'random' sobre N imagenes con K fijo."""
    print(f"\n=== Analisis 1: inicializacion de centroides (K={K}, N={n_imagenes}) ===")

    estrategias = ['first', 'random']
    resultados = {e: {'iteraciones': [], 'wcd': [], 'tiempo': []} for e in estrategias}

    np.random.seed(42)
    indices = np.random.choice(len(imagenes), n_imagenes, replace=False)

    for est in estrategias:
        for idx in indices:
            t0 = time.time()
            km = Kmeans.KMeans(imagenes[idx], K=K, options={'km_init': est})
            km.fit()
            elapsed = time.time() - t0
            resultados[est]['iteraciones'].append(km.num_iter)
            resultados[est]['wcd'].append(km.withinClassDistance())
            resultados[est]['tiempo'].append(elapsed)

    tabla = {}
    print(f"  {'Estrategia':<10}  {'Iter media':>10}  {'WCD media':>10}  {'t medio (ms)':>14}")
    for est in estrategias:
        iter_media = np.mean(resultados[est]['iteraciones'])
        wcd_media = np.mean(resultados[est]['wcd'])
        t_media_ms = np.mean(resultados[est]['tiempo']) * 1000
        print(f"  {est:<10}  {iter_media:>10.2f}  {wcd_media:>10.2f}  {t_media_ms:>14.2f}")
        tabla[est] = {
            'iter_media': float(iter_media),
            'wcd_media': float(wcd_media),
            't_medio_ms': float(t_media_ms),
        }

    return tabla


# ====================================================================
# Analisis 2 — Valor de K en KNN
# ====================================================================

def analisis_2_k_knn(train_imgs, train_class, test_imgs, test_class):
    """Mide accuracy del KNN para varios valores de k."""
    print("\n=== Analisis 2: valor de K en KNN ===")
    knn = KNN(train_imgs, train_class)
    tabla = {}
    print(f"  {'k':>3}  {'accuracy':>10}  {'tiempo (s)':>10}")
    for k in [1, 3, 5, 7, 9, 11]:
        t0 = time.time()
        pred = knn.predict(test_imgs, k=k)
        acc = medir_accuracy(pred, test_class)
        elapsed = time.time() - t0
        print(f"  {k:>3}  {acc:>9.2f}%  {elapsed:>10.2f}")
        tabla[k] = {'accuracy': float(acc), 'tiempo_s': float(elapsed)}
    return tabla


# ====================================================================
# Analisis 3 — Criterio de find_bestK
# ====================================================================

def analisis_3_criterio_bestK(imagenes, n_imagenes=30, max_K=8):
    """Compara distintos umbrales del criterio del decremento de WCD.

    Para cada imagen, calculamos las WCD desde K=2 hasta max_K, y vemos
    cual K elegiria cada umbral. Despues mostramos la K media y la
    distribucion para cada umbral.
    """
    print(f"\n=== Analisis 3: criterio de find_bestK (N={n_imagenes}) ===")

    np.random.seed(42)
    indices = np.random.choice(len(imagenes), n_imagenes, replace=False)
    umbrales = [10, 15, 20, 25, 30]

    K_elegidas = {u: [] for u in umbrales}

    for idx in indices:
        # Calculamos las WCD una sola vez por imagen (caro, no repetir)
        wcds = []
        for k in range(2, max_K + 1):
            km = Kmeans.KMeans(imagenes[idx], K=k)
            km.fit()
            wcds.append(km.withinClassDistance())

        # Para cada umbral, decidimos la K elegida
        for u in umbrales:
            K_u = 2  # por defecto cogemos K=2 si nada cumple
            for i in range(1, len(wcds)):
                porcentaje = 100 - 100 * (wcds[i] / wcds[i - 1])
                if porcentaje < u:
                    K_u = (i + 2) - 1  # K_i = i+2, retrocedemos 1
                    break
                K_u = i + 2  # avanzamos
            K_elegidas[u].append(K_u)

    tabla = {}
    print(f"  {'umbral %':>10}  {'K media':>10}  {'K min':>6}  {'K max':>6}")
    for u in umbrales:
        ks = K_elegidas[u]
        K_media = float(np.mean(ks))
        K_min = int(np.min(ks))
        K_max = int(np.max(ks))
        print(f"  {u:>10}  {K_media:>10.2f}  {K_min:>6}  {K_max:>6}")
        tabla[u] = {'K_media': K_media, 'K_min': K_min, 'K_max': K_max}

    return tabla


# ====================================================================
# Analisis 4 — Normalizacion Min-Max en el KNN
# ====================================================================

class KNN_normalizable:
    """KNN igual al original pero con opcion de normalizar Min-Max.

    En el Bloc 4 sesion 7 se vio que K-NN decide solo a partir de
    distancias y por eso la escala de las variables importa. Aqui
    comprobamos si normalizar los pixeles cambia el accuracy en este
    dataset.
    """
    def __init__(self, train_data, labels, normalizar=False):
        self.normalizar = normalizar
        self.train_data = self._procesar(train_data, ajustar_minmax=True)
        self.labels = np.array(labels)

    def _procesar(self, data, ajustar_minmax=False):
        # Igual que el KNN base: pasar a gris y aplanar
        data = data.astype(float)
        data = utils.rgb2gray(data)
        data = data.reshape(data.shape[0], -1)

        if self.normalizar:
            if ajustar_minmax:
                # Guardamos el min y max del train para reutilizarlos
                # con el test (el test se normaliza con el rango del train)
                self._min = data.min()
                self._max = data.max()
            # Normalizacion Min-Max: x' = (x - min) / (max - min)
            data = (data - self._min) / (self._max - self._min)
        return data

    def predict(self, test_data, k):
        test_proc = self._procesar(test_data, ajustar_minmax=False)
        n_test = test_proc.shape[0]
        predicciones = np.empty(n_test, dtype=self.labels.dtype)
        for i in range(n_test):
            diferencias = self.train_data - test_proc[i]
            distancias = np.sqrt(np.sum(diferencias ** 2, axis=1))
            k_idx = np.argsort(distancias)[:k]
            vecinos = self.labels[k_idx]
            clases, votos = np.unique(vecinos, return_counts=True)
            predicciones[i] = clases[np.argmax(votos)]
        return predicciones


def analisis_4_normalizacion(train_imgs, train_class, test_imgs, test_class, k=3):
    """Compara accuracy del KNN con y sin normalizacion Min-Max."""
    print(f"\n=== Analisis 4: normalizacion Min-Max en el KNN (k={k}) ===")
    tabla = {}
    print(f"  {'configuracion':<20}  {'accuracy':>10}  {'tiempo (s)':>10}")
    for normalizar in [False, True]:
        t0 = time.time()
        knn = KNN_normalizable(train_imgs, train_class, normalizar=normalizar)
        pred = knn.predict(test_imgs, k=k)
        acc = medir_accuracy(pred, test_class)
        elapsed = time.time() - t0
        nombre = 'con Min-Max' if normalizar else 'sin normalizar'
        print(f"  {nombre:<20}  {acc:>9.2f}%  {elapsed:>10.2f}")
        tabla[nombre] = {'accuracy': float(acc), 'tiempo_s': float(elapsed)}
    return tabla


# ====================================================================
# Main
# ====================================================================

def main():
    print("Cargando dataset...")
    train_imgs, train_class, _, test_imgs, test_class, _ = read_dataset(
        root_folder=IMAGES_DIR + os.sep, gt_json=GT_PATH
    )

    resultados = {}
    resultados['analisis_1_inicializacion'] = analisis_1_inicializacion(train_imgs)
    resultados['analisis_2_k_knn'] = analisis_2_k_knn(train_imgs, train_class, test_imgs, test_class)
    resultados['analisis_3_criterio_bestK'] = analisis_3_criterio_bestK(train_imgs)
    resultados['analisis_4_normalizacion'] = analisis_4_normalizacion(
        train_imgs, train_class, test_imgs, test_class
    )

    out_path = RESULTS_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f"\nResultados guardados en {out_path}")


if __name__ == '__main__':
    main()
