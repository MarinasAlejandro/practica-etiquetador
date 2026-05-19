"""Prueba rapida del KMeans sobre una imagen real.

Carga una imagen del dataset, aplica KMeans con varios K, comprueba
que find_bestK encuentra una K razonable y que get_colors devuelve los
nombres de los colores predominantes.
"""

import os
import sys
import time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Kmeans import KMeans, get_colors


def cargar_imagen(path, w=60, h=80):
    """Carga una imagen y la redimensiona al tamano del dataset (60x80)."""
    img = Image.open(path).convert('RGB')
    img = img.resize((w, h))
    return np.array(img)


def main():
    # Cogemos una imagen cualquiera del train
    img_path = './images/train/10004.jpg'
    img = cargar_imagen(img_path)
    print(f"Imagen: {img_path}")
    print(f"  shape: {img.shape}")
    print(f"  rango RGB: {img.min()} - {img.max()}")

    print("\n--- Test 1: KMeans con K=3 ('first') ---")
    t0 = time.time()
    km = KMeans(img, K=3)
    km.fit()
    print(f"  iteraciones hasta convergencia: {km.num_iter}")
    print(f"  tiempo: {time.time() - t0:.2f}s")
    print(f"  centroides RGB:\n{km.centroids.astype(int)}")
    print(f"  WCD: {km.withinClassDistance():.2f}")
    print(f"  colores detectados: {get_colors(km.centroids)}")

    print("\n--- Test 2: KMeans con K=3 ('random') ---")
    np.random.seed(42)
    km2 = KMeans(img, K=3, options={'km_init': 'random'})
    km2.fit()
    print(f"  iteraciones: {km2.num_iter}")
    print(f"  centroides RGB:\n{km2.centroids.astype(int)}")
    print(f"  colores detectados: {get_colors(km2.centroids)}")

    print("\n--- Test 3: find_bestK (max_K=8) ---")
    t0 = time.time()
    km3 = KMeans(img, K=2)
    mejor_K = km3.find_bestK(max_K=8)
    print(f"  K optima encontrada: {mejor_K}")
    print(f"  tiempo: {time.time() - t0:.2f}s")
    print(f"  colores con K={mejor_K}: {get_colors(km3.centroids)}")

    print("\n--- Test 4: tabla de WCD por K (replica el ejemplo del PDF) ---")
    print(f"  {'K':>3}  {'WCD':>10}  {'100-%DEC':>10}")
    wcd_anterior = None
    for k in range(2, 7):
        km4 = KMeans(img, K=k)
        km4.fit()
        wcd = km4.withinClassDistance()
        if wcd_anterior is None:
            print(f"  {k:>3}  {wcd:>10.2f}  {'-':>10}")
        else:
            dec = 100 - 100 * (wcd / wcd_anterior)
            print(f"  {k:>3}  {wcd:>10.2f}  {dec:>9.2f}%")
        wcd_anterior = wcd


if __name__ == '__main__':
    main()
