"""Etiquetador de imagenes y buscador combinado.

Junta los dos modelos (KNN para la forma, KMeans para el color) y permite:
  1. Etiquetar una imagen nueva (forma + colores predominantes).
  2. Buscar en el dataset las imagenes que cumplan unos filtros.
"""

import numpy as np

import KNN as KNN_module
import Kmeans
import utils


# Mapeo de palabras del usuario a las etiquetas de forma del dataset.
# El dataset usa formas en plural ("Dresses", "Shirts"...) pero el usuario
# normalmente escribe en singular ("dress", "shirt"). Este dict permite
# entender ambos.
SHAPE_SYNONYMS = {
    'dress': 'Dresses', 'dresses': 'Dresses',
    'shirt': 'Shirts', 'shirts': 'Shirts',
    'jean': 'Jeans', 'jeans': 'Jeans', 'pants': 'Jeans',
    'short': 'Shorts', 'shorts': 'Shorts',
    'sock': 'Socks', 'socks': 'Socks',
    'sandal': 'Sandals', 'sandals': 'Sandals',
    'handbag': 'Handbags', 'handbags': 'Handbags',
    'bag': 'Handbags', 'bags': 'Handbags',
    'heel': 'Heels', 'heels': 'Heels',
    'flipflop': 'Flip Flops', 'flipflops': 'Flip Flops',
    'flip-flop': 'Flip Flops', 'flip-flops': 'Flip Flops',
}


def _normalizar_color(color):
    """Pone el color con la primera letra en mayuscula (formato del dataset)."""
    return color.strip().capitalize()


def _normalizar_forma(forma):
    """Convierte la palabra del usuario a la etiqueta del dataset."""
    clave = forma.strip().lower().replace(' ', '')
    return SHAPE_SYNONYMS.get(clave, forma.strip().title())


def retrieval_by_color(images, color_labels, query_color):
    """Devuelve las imagenes cuyas etiquetas de color contienen query_color.

    Las etiquetas de color en el dataset son listas (una imagen puede ser
    "Pink y White" a la vez), asi que comprobamos pertenencia a la lista.
    """
    query = _normalizar_color(query_color)
    indices = [i for i, etiquetas in enumerate(color_labels) if query in etiquetas]
    return images[indices], indices


def retrieval_by_shape(images, shape_labels, query_shape):
    """Devuelve las imagenes cuya etiqueta de forma coincide con query_shape."""
    query = _normalizar_forma(query_shape)
    indices = [i for i, etiqueta in enumerate(shape_labels) if etiqueta == query]
    return images[indices], indices


def retrieval_combined(images, color_labels, shape_labels, query_color=None, query_shape=None):
    """Devuelve las imagenes que cumplen color Y forma a la vez.

    Si solo se pasa uno de los dos filtros, equivale al retrieval simple.
    Si se pasan los dos, hacemos AND: la imagen tiene que cumplir ambas
    condiciones para aparecer en el resultado.
    """
    n = len(images)
    mascara = np.ones(n, dtype=bool)

    if query_shape is not None:
        forma = _normalizar_forma(query_shape)
        mascara &= np.array([etq == forma for etq in shape_labels])

    if query_color is not None:
        color = _normalizar_color(query_color)
        mascara &= np.array([color in etqs for etqs in color_labels])

    indices = np.where(mascara)[0].tolist()
    return images[indices], indices


def predict_image(image, knn_model, K_color=None, max_K=8, options_kmeans=None):
    """Etiqueta una imagen suelta con forma (KNN) y colores (KMeans).

    Args:
        image: array (alto, ancho, 3) en RGB.
        knn_model: instancia de KNN ya entrenada.
        K_color: numero de colores a buscar. Si None, lo decide find_bestK.
        max_K: maximo de colores a probar si K_color es None.
        options_kmeans: dict de opciones para KMeans.

    Returns:
        dict con:
          - 'shape': forma predicha (string).
          - 'colors': lista de colores predominantes (sin repetir).
          - 'K': numero de colores efectivamente usado.
    """
    # 1. Forma con KNN. El predict espera un batch de imagenes.
    forma = knn_model.predict(image[np.newaxis, ...], k=3)[0]

    # 2. Colores con KMeans. Si no se da K, lo encontramos automaticamente.
    if K_color is None:
        km = Kmeans.KMeans(image, K=2, options=options_kmeans)
        km.find_bestK(max_K=max_K)
    else:
        km = Kmeans.KMeans(image, K=K_color, options=options_kmeans)
        km.fit()

    colores = Kmeans.get_colors(km.centroids)
    # Quitamos colores repetidos manteniendo el orden de aparicion
    colores_unicos = list(dict.fromkeys(colores))

    return {
        'shape': str(forma),
        'colors': colores_unicos,
        'K': km.K,
    }


def label_dataset(images, knn_model, K_color=None, max_K=8, verbose=True):
    """Etiqueta todo un conjunto de imagenes con predict_image.

    Devuelve una lista de dicts (uno por imagen) con la forma y los
    colores predichos por nuestro sistema. Util para pre-etiquetar el
    dataset y que el cercador opere sobre etiquetas predichas, no sobre
    el ground truth.
    """
    etiquetas = []
    n = len(images)
    for i, img in enumerate(images):
        if verbose and i % 100 == 0:
            print(f"  Etiquetando {i}/{n}...")
        etiquetas.append(predict_image(img, knn_model, K_color=K_color, max_K=max_K))
    return etiquetas
