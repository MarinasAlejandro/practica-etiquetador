[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/IK-jwSgP)

# Practica Etiquetador

Agente de etiquetado automatico de imagenes de ropa para hacer busquedas en lenguaje natural en una tienda online.

El sistema asigna a cada imagen dos tipos de etiquetas:

- **Forma** (Dresses, Shirts, Heels, Flip Flops, Shorts, Jeans, Socks, Sandals, Handbags) usando **KNN** (aprendizaje supervisado).
- **Color** (11 colores basicos: Red, Orange, Brown, Yellow, Green, Blue, Purple, Pink, Black, Grey, White) usando **K-means** (aprendizaje no supervisado).

Ademas incluye un **buscador** que combina ambas etiquetas para resolver consultas tipo `"pink dress"`.

## Estructura

```
.
├── KNN.py              # Clasificador K-Nearest Neighbours (forma)
├── Kmeans.py           # Clustering K-Means (color)
├── my_labeling.py      # Etiquetador end-to-end y buscador combinado
├── utils.py            # Utilidades del template (rgb2gray, get_color_prob)
├── utils_data.py       # Utilidades del template (read_dataset, visualize_*)
├── images/             # Dataset de entrenamiento y test
├── informe/            # Informe (Desenvolupament + Analisis d'eficiencia)
├── presentacion/       # Slides para la presentacion oral
└── app/                # Frontend Flask (upload + buscador web)
```

## Como ejecutar

```bash
# 1. Crear entorno e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Probar el KNN
python test_knn.py

# 3. Lanzar el frontend web
python app/app.py
```

## Autor

Alejandro Marinas — Aprenentatge Automatic, La Salle Gracia.
