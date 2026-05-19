"""Prueba rapida del etiquetador end-to-end y del cercador.

1. Entrena el KNN con el train.
2. Coge una imagen aleatoria y la etiqueta con predict_image.
3. Prueba retrievals (por color, por forma, combinado) sobre el ground truth.
"""

import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from KNN import KNN
from my_labeling import (
    predict_image,
    retrieval_by_color,
    retrieval_by_shape,
    retrieval_combined,
)
from utils_data import read_dataset


def main():
    print("Cargando dataset...")
    train_imgs, train_class, _, test_imgs, test_class, test_color = read_dataset(
        root_folder='./images/', gt_json='./images/gt.json'
    )

    print("Entrenando KNN...")
    t0 = time.time()
    knn = KNN(train_imgs, train_class)
    print(f"  hecho en {time.time() - t0:.2f}s")

    print("\n--- Test 1: predict_image sobre imagenes aleatorias ---")
    np.random.seed(0)
    for idx in np.random.choice(len(test_imgs), 5, replace=False):
        img = test_imgs[idx]
        resultado = predict_image(img, knn)
        forma_real = test_class[idx]
        colores_reales = test_color[idx]
        ok_forma = '✓' if resultado['shape'] == forma_real else '✗'
        print(f"  img {idx}:")
        print(f"    predicho: forma={resultado['shape']}, colores={resultado['colors']} (K={resultado['K']})")
        print(f"    real:     forma={forma_real}, colores={colores_reales}")
        print(f"    forma {ok_forma}")

    print("\n--- Test 2: retrieval_by_color ('Pink') ---")
    imgs_pink, _ = retrieval_by_color(test_imgs, test_color, 'Pink')
    print(f"  Encontradas {len(imgs_pink)} imagenes con color Pink")

    print("\n--- Test 3: retrieval_by_shape ('Dresses') ---")
    imgs_dress, _ = retrieval_by_shape(test_imgs, test_class, 'Dresses')
    print(f"  Encontradas {len(imgs_dress)} imagenes con forma Dresses")

    print("\n--- Test 4: retrieval_combined ('pink', 'dress') ---")
    imgs_pd, idx_pd = retrieval_combined(test_imgs, test_color, test_class, 'pink', 'dress')
    print(f"  Encontradas {len(imgs_pd)} imagenes que son vestidos rosas")
    if len(idx_pd) > 0:
        print(f"  Ejemplos de etiquetas reales:")
        for i in idx_pd[:5]:
            print(f"    forma={test_class[i]}, colores={test_color[i]}")


if __name__ == '__main__':
    main()
