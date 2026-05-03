import numpy as np
import utils


class KMeans:
    """Clustering K-Means para detectar los colores predominantes de una imagen.

    Cada pixel se trata como un punto en el espacio RGB de 3 dimensiones.
    El algoritmo busca K centroides que agrupen los pixeles segun su color.
    Una vez convergido, cada centroide representa un color predominante.
    """

    def __init__(self, X, K=1, options=None):
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)

    def _init_X(self, X):
        # Pendiente: reformatear a (N_pixels, 3)
        pass

    def _init_options(self, options=None):
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'tolerance' not in options:
            options['tolerance'] = 0.0
        if 'max_iter' not in options:
            options['max_iter'] = 100
        if 'fitting' not in options:
            options['fitting'] = 'WCD'
        self.options = options

    def _init_centroids(self):
        # Pendiente
        pass

    def get_labels(self):
        # Pendiente
        pass

    def get_centroids(self):
        # Pendiente
        pass

    def converges(self):
        # Pendiente
        pass

    def fit(self):
        # Pendiente
        pass

    def withinClassDistance(self):
        # Pendiente
        pass

    def find_bestK(self, max_K):
        # Pendiente
        pass


def get_colors(centroids):
    """Devuelve el color (string) mas probable para cada centroide RGB."""
    # Pendiente
    pass


def distance(X, C):
    """Distancia euclidiana entre cada punto de X y cada centroide de C."""
    # Pendiente
    pass
