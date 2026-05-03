import numpy as np
import utils


class KNN:
    """Clasificador K-Nearest Neighbours para predecir la forma de una prenda.

    Convierte las imagenes de entrenamiento a escala de grises y las aplana
    a vectores de pixeles. Para predecir, calcula la distancia euclidiana
    entre cada imagen nueva y todas las del entrenamiento, coge los k vecinos
    mas cercanos y vota la clase mayoritaria.
    """

    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        """Pasa las imagenes a gris y las aplana a vectores de N pixeles.

        Si la imagen es 80x60x3 (RGB) acaba en un vector de 4800 dimensiones.
        Guarda el resultado en self.train_data como (n_imagenes, n_pixeles).
        """
        train_data = train_data.astype(float)
        # Si vienen en color (4 dimensiones), las pasamos a gris
        if train_data.ndim == 4:
            train_data = utils.rgb2gray(train_data)
        # Aplanamos cada imagen: cada fila es una imagen, cada columna un pixel
        n_imagenes = train_data.shape[0]
        self.train_data = train_data.reshape(n_imagenes, -1)

    def get_k_neighbours(self, test_data, k):
        """Calcula los k vecinos mas cercanos para cada imagen de test.

        Para cada imagen de test calcula la distancia euclidiana a todas
        las del entrenamiento, ordena, coge las k mas cercanas y guarda
        sus etiquetas en self.neighbors (forma (n_test, k)).
        """
        # Preprocesamos el test exactamente igual que el train
        test_data = test_data.astype(float)
        if test_data.ndim == 4:
            test_data = utils.rgb2gray(test_data)
        n_test = test_data.shape[0]
        test_flat = test_data.reshape(n_test, -1)

        # Para cada imagen de test, calculamos distancias a todo el train
        # Hacemos un bucle por imagen de test (claridad) pero vectorizamos
        # el calculo de distancias respecto a todas las del entrenamiento
        self.neighbors = np.empty((n_test, k), dtype=self.labels.dtype)
        for i in range(n_test):
            diferencias = self.train_data - test_flat[i]
            distancias = np.sqrt(np.sum(diferencias ** 2, axis=1))
            # Indices de los k vecinos mas cercanos (menor distancia primero)
            k_idx = np.argsort(distancias)[:k]
            self.neighbors[i] = self.labels[k_idx]

    def get_class(self):
        """Devuelve la clase mayoritaria entre los k vecinos de cada imagen.

        En caso de empate se queda con la primera por orden alfabetico
        (comportamiento por defecto de np.unique).
        """
        predicciones = []
        for vecinos in self.neighbors:
            clases, votos = np.unique(vecinos, return_counts=True)
            predicciones.append(clases[np.argmax(votos)])
        return np.array(predicciones)

    def predict(self, test_data, k):
        """Predice la clase de cada imagen de test (forma de la prenda)."""
        self.get_k_neighbours(test_data, k)
        return self.get_class()
