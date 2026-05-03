"""Etiquetador de imagenes y buscador combinado.

Junta los dos modelos (KNN para forma, KMeans para color) y permite:
  1. Etiquetar una imagen nueva (forma + colores predominantes).
  2. Buscar en el dataset las imagenes que cumplan unos filtros.
"""

import numpy as np
import KNN
import Kmeans
import utils
import utils_data


def predict_image(image, knn_model, K_color=3):
    """Etiqueta una imagen suelta con forma (KNN) y colores (KMeans).

    Devuelve un dict con la forma predicha y la lista de colores predominantes.
    """
    # Pendiente
    pass


def retrieval_by_color(images, color_labels, query_color):
    """Devuelve las imagenes cuyas etiquetas de color contienen query_color."""
    # Pendiente
    pass


def retrieval_by_shape(images, shape_labels, query_shape):
    """Devuelve las imagenes cuya etiqueta de forma coincide con query_shape."""
    # Pendiente
    pass


def retrieval_combined(images, color_labels, shape_labels, query_color, query_shape):
    """Devuelve las imagenes que cumplen tanto el color como la forma indicados."""
    # Pendiente
    pass
