"""Pre-etiqueta todo el dataset con NUESTROS modelos y guarda el resultado.

El frontend usara este JSON para que el cercador opere sobre etiquetas
PREDICHAS por nuestro sistema (no sobre el ground truth). Asi se demuestra
que el etiquetador funciona en un escenario real: la tienda recibe un
producto nuevo, lo procesa y se puede buscar.

Salida: predicted_labels.json con la forma:
{
  "train/12345": {"shape": "Dresses", "colors": ["Pink", "White"], "K": 5},
  ...
}
"""

import json
import os
import time
import numpy as np
from PIL import Image

from KNN import KNN
import Kmeans


def cargar_imagenes_con_nombres(root_folder, gt_path, split, w=60, h=80):
    """Carga las imagenes de un split (train/test) junto con sus nombres."""
    gt = json.load(open(gt_path))
    nombres = list(gt[split].keys())
    formas = [gt[split][n][0] for n in nombres]

    imagenes = []
    for nombre in nombres:
        path = os.path.join(root_folder, split, f"{nombre}.jpg")
        img = Image.open(path).convert('RGB').resize((w, h))
        imagenes.append(np.array(img))
    return np.array(imagenes), nombres, np.array(formas)


def main():
    print("=== Pre-etiquetado del dataset ===\n")

    # 1. Cargar train + test
    print("Cargando imagenes del dataset...")
    t0 = time.time()
    train_imgs, train_names, train_class = cargar_imagenes_con_nombres(
        './images', './images/gt.json', 'train'
    )
    test_imgs, test_names, test_class = cargar_imagenes_con_nombres(
        './images', './images/gt.json', 'test'
    )
    print(f"  train: {len(train_imgs)} imagenes")
    print(f"  test:  {len(test_imgs)} imagenes")
    print(f"  carga: {time.time() - t0:.1f}s\n")

    # 2. Entrenar KNN una sola vez
    print("Entrenando KNN con el train...")
    t0 = time.time()
    knn = KNN(train_imgs, train_class)
    print(f"  hecho en {time.time() - t0:.1f}s\n")

    # 3. Predecir formas en batch (mucho mas rapido que una a una)
    print("Prediciendo formas en batch (k=3)...")
    t0 = time.time()
    formas_train = knn.predict(train_imgs, k=3)
    formas_test = knn.predict(test_imgs, k=3)
    print(f"  hecho en {time.time() - t0:.1f}s\n")

    # 4. Predecir colores para cada imagen con KMeans (K=5 fijo)
    print("Detectando colores con KMeans (K=5 por imagen)...")
    K_color = 5
    resultado = {}
    todos = (
        list(zip(train_imgs, train_names, formas_train, ['train'] * len(train_imgs))) +
        list(zip(test_imgs, test_names, formas_test, ['test'] * len(test_imgs)))
    )

    t0 = time.time()
    for i, (img, nombre, forma, split) in enumerate(todos):
        if i % 200 == 0:
            transcurrido = time.time() - t0
            if i > 0:
                eta = transcurrido / i * (len(todos) - i)
                print(f"  {i}/{len(todos)} ({transcurrido:.0f}s, ETA {eta:.0f}s)")
            else:
                print(f"  {i}/{len(todos)}")

        km = Kmeans.KMeans(img, K=K_color)
        km.fit()
        colores = Kmeans.get_colors(km.centroids)
        # Quitamos colores repetidos manteniendo el orden de aparicion
        colores_unicos = list(dict.fromkeys(colores))

        clave = f"{split}/{nombre}"
        resultado[clave] = {
            'shape': str(forma),
            'colors': colores_unicos,
            'K': K_color,
        }
    print(f"  total: {time.time() - t0:.0f}s\n")

    # 5. Guardar
    out = './predicted_labels.json'
    with open(out, 'w') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"Guardado en {out} ({len(resultado)} imagenes etiquetadas)")


if __name__ == '__main__':
    main()
