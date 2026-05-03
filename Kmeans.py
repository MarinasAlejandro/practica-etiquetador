import numpy as np
import utils


class KMeans:
    """Clustering K-Means para detectar los colores predominantes de una imagen.

    Cada pixel se trata como un punto en el espacio RGB (3 dimensiones).
    El algoritmo busca K centroides que agrupen los pixeles segun su color.
    Una vez convergido, cada centroide representa un color predominante.
    """

    def __init__(self, X, K=1, options=None):
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)

    def _init_X(self, X):
        """Reformatea la imagen a una matriz (N_pixeles, 3).

        Si la imagen viene como (alto, ancho, 3), aplanamos para que cada
        pixel sea una fila (un punto en el espacio RGB). Si ya viene como
        (N_pixeles, 3) la dejamos igual.
        """
        X = np.array(X, dtype=float)
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        self.X = X

    def _init_options(self, options=None):
        """Aplica las opciones por defecto si no se especifican."""
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
        """Inicializa K centroides segun la estrategia elegida.

        - 'first': los K primeros pixeles distintos de la imagen.
        - 'random': K pixeles aleatorios distintos.
        """
        estrategia = self.options['km_init']
        if estrategia == 'first':
            # Cogemos los K primeros pixeles distintos (sin repetir colores)
            centroides = []
            vistos = set()
            for pixel in self.X:
                clave = tuple(pixel)
                if clave not in vistos:
                    vistos.add(clave)
                    centroides.append(pixel)
                    if len(centroides) == self.K:
                        break
            self.centroids = np.array(centroides, dtype=float)
        elif estrategia == 'random':
            # K pixeles aleatorios sin repeticion
            indices = np.random.choice(len(self.X), self.K, replace=False)
            self.centroids = self.X[indices].astype(float)
        else:
            raise ValueError(f"Estrategia de inicializacion desconocida: {estrategia}")

        # Guardamos una copia inicial para comparar al comprobar convergencia
        self.old_centroids = np.zeros_like(self.centroids)

    def get_labels(self):
        """Asigna cada pixel al centroide mas cercano (distancia euclidiana)."""
        distancias = distance(self.X, self.centroids)
        self.labels = np.argmin(distancias, axis=1)

    def get_centroids(self):
        """Recalcula los centroides como la media de los pixeles de cada cluster."""
        self.old_centroids = self.centroids.copy()
        nuevos = np.zeros_like(self.centroids)
        for k in range(self.K):
            puntos_cluster = self.X[self.labels == k]
            if len(puntos_cluster) > 0:
                nuevos[k] = puntos_cluster.mean(axis=0)
            else:
                # Si un cluster se queda vacio, mantenemos el centroide anterior
                nuevos[k] = self.centroids[k]
        self.centroids = nuevos

    def converges(self):
        """True si los centroides ya no se mueven (dentro de la tolerancia)."""
        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])

    def fit(self):
        """Ejecuta el algoritmo: inicializa, asigna, recalcula y repite."""
        self._init_centroids()
        self.num_iter = 0
        for _ in range(self.options['max_iter']):
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            if self.converges():
                break

    def withinClassDistance(self):
        """Distancia intra-clase (WCD): media de las distancias al cuadrado
        de cada pixel a su centroide. Mide cuan compactos son los clusters.
        """
        distancias_2 = np.sum((self.X - self.centroids[self.labels]) ** 2, axis=1)
        return distancias_2.mean()

    def find_bestK(self, max_K):
        """Encuentra la K optima usando el criterio del 20% sobre la WCD.

        Probamos K = 2, 3, ..., max_K y nos quedamos con el primer K cuyo
        decremento sea menor del 20% respecto al K anterior. Es decir,
        paramos cuando anadir un cluster mas ya no compensa.
        Formula: 100 - 100 * WCD_k / WCD_{k-1} < 20
        """
        wcd_anterior = None
        mejor_K = 2
        for k in range(2, max_K + 1):
            self.K = k
            self.fit()
            wcd = self.withinClassDistance()
            if wcd_anterior is not None:
                porcentaje = 100 - 100 * (wcd / wcd_anterior)
                if porcentaje < 20:
                    mejor_K = k - 1
                    break
                mejor_K = k
            wcd_anterior = wcd
        self.K = mejor_K
        # Reentrenamos con el K final para que el modelo quede consistente
        self.fit()
        return mejor_K


def distance(X, C):
    """Distancia euclidiana entre cada punto de X y cada centroide de C.

    Devuelve una matriz (N, K) donde N es el numero de puntos y K el de
    centroides. Usamos broadcasting de numpy para evitar un bucle.
    """
    # X: (N, 3) -> (N, 1, 3); C: (K, 3) -> (1, K, 3); diff: (N, K, 3)
    diferencias = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    return np.sqrt(np.sum(diferencias ** 2, axis=2))


def get_colors(centroids):
    """Devuelve el nombre del color mas probable para cada centroide RGB.

    Llama a utils.get_color_prob, que para cada centroide devuelve un vector
    de 11 probabilidades (una por cada color basico universal). Cogemos el
    argmax y lo mapeamos al nombre con utils.colors.
    """
    # get_color_prob espera una imagen, asi que pasamos los centroides
    # como una "fila" de pixeles para que los procese de golpe
    centroides_imagen = centroids.reshape(1, -1, 3)
    probabilidades = utils.get_color_prob(centroides_imagen)
    # probabilidades tiene shape (1, K, 11): aplanamos a (K, 11)
    probabilidades = probabilidades.reshape(centroids.shape[0], -1)
    indices_color = np.argmax(probabilidades, axis=1)
    return utils.colors[indices_color].tolist()
