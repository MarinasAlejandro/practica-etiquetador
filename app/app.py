"""Frontend Flask del etiquetador.

Dos funcionalidades, como pidio el profesor:

1. POST /predict: subes una imagen y te dice que prenda es y de que color.
2. GET  /search:  filtras por color y forma y te muestra las imagenes
   coincidentes del dataset (pre-etiquetadas con NUESTRO sistema).

Para arrancar la app, KNN se entrena al inicio (tarda <1s con el dataset
real) y el JSON de etiquetas predichas se carga en memoria.
"""

import json
import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

# Anadimos el directorio raiz al path para poder importar KNN, Kmeans, etc.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from KNN import KNN
import Kmeans
import my_labeling
import utils
from utils_data import read_dataset


app = Flask(__name__, template_folder='templates', static_folder='static')

# Limites razonables para imagenes subidas
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

# === Estado global cargado al arrancar ===

KNN_MODEL = None
PREDICTED_LABELS = {}  # {'train/12345': {'shape': ..., 'colors': [...]}}
COLORS_AVAILABLE = sorted(utils.colors.tolist())
SHAPES_AVAILABLE = sorted([
    'Dresses', 'Shirts', 'Heels', 'Flip Flops',
    'Shorts', 'Jeans', 'Socks', 'Sandals', 'Handbags',
])


def cargar_knn():
    """Entrena el KNN con el train del dataset."""
    print("[app] Cargando dataset y entrenando KNN...")
    train_imgs, train_class, _, _, _, _ = read_dataset(
        root_folder=os.path.join(ROOT_DIR, 'images') + '/',
        gt_json=os.path.join(ROOT_DIR, 'images/gt.json'),
    )
    return KNN(train_imgs, train_class)


def cargar_etiquetas_predichas():
    """Carga las etiquetas predichas pre-calculadas (si existen)."""
    path = os.path.join(ROOT_DIR, 'predicted_labels.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"[app] AVISO: {path} no existe. El buscador devolvera vacio.")
    print(f"      Ejecuta primero: python preprocess_dataset.py")
    return {}


# === Rutas ===

@app.route('/')
def index():
    """Home con las dos funcionalidades."""
    return render_template(
        'index.html',
        colors=COLORS_AVAILABLE,
        shapes=SHAPES_AVAILABLE,
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Etiqueta la imagen subida con forma y colores predominantes."""
    if 'image' not in request.files:
        return jsonify({'error': 'No se ha subido ninguna imagen'}), 400

    archivo = request.files['image']
    if archivo.filename == '':
        return jsonify({'error': 'Nombre de archivo vacio'}), 400

    # Cargamos la imagen y la pasamos al tamano del dataset (60x80)
    try:
        img = Image.open(archivo.stream).convert('RGB').resize((60, 80))
    except Exception as e:
        return jsonify({'error': f'No se pudo abrir la imagen: {e}'}), 400

    img_array = np.array(img)

    # Aplicamos el etiquetador end-to-end
    resultado = my_labeling.predict_image(img_array, KNN_MODEL)
    return jsonify(resultado)


@app.route('/search')
def search():
    """Busca imagenes del dataset pre-etiquetadas que cumplan los filtros.

    Acepta tanto color y/o forma. Si no se pasa ninguno, devuelve vacio
    para no inundar al usuario con todo el dataset.
    """
    color = request.args.get('color', '').strip()
    forma = request.args.get('shape', '').strip()
    limite = int(request.args.get('limit', 24))

    if not color and not forma:
        return jsonify({'results': [], 'total': 0})

    color_norm = color.capitalize() if color else None
    forma_norm = my_labeling._normalizar_forma(forma) if forma else None

    resultados = []
    for clave, etqs in PREDICTED_LABELS.items():
        cumple_color = (color_norm is None) or (color_norm in etqs['colors'])
        cumple_forma = (forma_norm is None) or (etqs['shape'] == forma_norm)
        if cumple_color and cumple_forma:
            resultados.append({
                'path': f'/dataset-image/{clave}',
                'shape': etqs['shape'],
                'colors': etqs['colors'],
            })

    total = len(resultados)
    resultados = resultados[:limite]
    return jsonify({'results': resultados, 'total': total, 'shown': len(resultados)})


@app.route('/dataset-image/<split>/<name>')
def dataset_image(split, name):
    """Sirve las imagenes del dataset (train/test) al frontend."""
    if split not in ('train', 'test'):
        return 'Split invalido', 400
    folder = os.path.join(ROOT_DIR, 'images', split)
    return send_from_directory(folder, f'{name}.jpg')


# === Inicializacion ===

KNN_MODEL = cargar_knn()
PREDICTED_LABELS = cargar_etiquetas_predichas()
print(f"[app] KNN listo. {len(PREDICTED_LABELS)} imagenes pre-etiquetadas en memoria.")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
