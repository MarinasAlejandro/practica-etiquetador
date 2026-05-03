"""Prueba rapida del KNN sobre el dataset real.

Carga el dataset completo, entrena el KNN y mide el accuracy con varios
valores de k. Sirve para verificar que la implementacion funciona y para
hacernos una idea del rendimiento del modelo.
"""

import time
import numpy as np

from KNN import KNN
from utils_data import read_dataset


def medir_accuracy(predicciones, etiquetas_reales):
    """Devuelve el porcentaje de aciertos."""
    aciertos = np.sum(predicciones == etiquetas_reales)
    return 100.0 * aciertos / len(etiquetas_reales)


def main():
    print("Cargando dataset...")
    train_imgs, train_class, _, test_imgs, test_class, _ = read_dataset(
        root_folder='./images/', gt_json='./images/gt.json'
    )
    print(f"  train: {train_imgs.shape}, etiquetas: {train_class.shape}")
    print(f"  test:  {test_imgs.shape}, etiquetas: {test_class.shape}")
    print(f"  clases: {sorted(set(train_class))}")

    print("\nEntrenando KNN (preprocesando imagenes a gris)...")
    t0 = time.time()
    knn = KNN(train_imgs, train_class)
    print(f"  hecho en {time.time() - t0:.2f}s, shape train procesado: {knn.train_data.shape}")

    print("\nResultados con varios valores de k:")
    for k in [1, 3, 5, 7, 9]:
        t0 = time.time()
        predicciones = knn.predict(test_imgs, k)
        acc = medir_accuracy(predicciones, test_class)
        print(f"  k={k:2d}  ->  accuracy = {acc:5.2f}%  ({time.time() - t0:.1f}s)")


if __name__ == '__main__':
    main()
