[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/IK-jwSgP)

# Practica Etiquetador

Agente de etiquetado automatico de imagenes de ropa para una tienda online: dada una
imagen, predice la **forma** de la prenda (KNN supervisado) y los **colores predominantes**
(K-Means no supervisado). Sobre estas etiquetas se construye un **buscador combinado**
con interfaz web que permite consultas tipo `"pink dress"`.

- **Forma** (9 clases): Dresses, Shirts, Heels, Flip Flops, Shorts, Jeans, Socks, Sandals, Handbags.
- **Color** (11 colores basicos): Red, Orange, Brown, Yellow, Green, Blue, Purple, Pink, Black, Grey, White.

## Estructura del repo

```
.
├── KNN.py                    # Clasificador K-Nearest Neighbours (forma)
├── Kmeans.py                 # Clustering K-Means (color)
├── my_labeling.py            # Etiquetador end-to-end, buscador y parse_query_text
├── analysis.py               # 4 analisis de eficiencia (genera informe/results.json)
├── analyze_duplicates.py     # Analisis metodologico: duplicados train/test (extra)
├── preprocess_dataset.py     # Pre-etiqueta el dataset con nuestros modelos
├── predicted_labels.json     # Etiquetas predichas para train+test (3.179 imagenes)
├── utils.py                  # Utilidades del template (rgb2gray, get_color_prob)
├── utils_data.py             # Utilidades del template (read_dataset, visualize_*)
├── test_knn.py               # Script de prueba del KNN
├── test_kmeans.py            # Script de prueba del KMeans
├── test_labeling.py          # Script de prueba del etiquetador y retrievals
├── test_extras.py            # Tests del parser textual, /search y hashing
├── images/                   # Dataset de entrenamiento y test
│   ├── gt.json               # Ground truth (etiquetas reales)
│   ├── gt_reduced.json       # Ground truth extendido con coordenadas
│   ├── train/                # 3.750 imagenes (2.328 etiquetadas)
│   └── test/                 # 1.394 imagenes (851 etiquetadas)
├── informe/                  # Informe + datos de los analisis + graficas
│   ├── informe.md            # Documento principal
│   ├── results.json          # Datos de los 4 analisis
│   ├── duplicates_analysis.json   # Resultado del analisis de duplicados
│   ├── generar_graficas.py   # Genera las figuras a partir de los JSON
│   └── figures/              # PNGs que se insertan en el informe
└── app/                      # Frontend Flask
    ├── app.py                # Backend con endpoints /predict y /search
    ├── templates/index.html  # Interfaz web (incluye buscador textual)
    └── static/               # CSS y JS
```

## Como instalar

```bash
# Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como ejecutar

### 1. Probar los modelos por separado

```bash
python test_knn.py        # Accuracy del KNN con varios k
python test_kmeans.py     # KMeans + tabla WCD por K
python test_labeling.py   # predict_image + retrievals
python test_extras.py     # Parser textual, /search y hash de duplicados
```

### 2. Generar las etiquetas predichas para el frontend

Solo hay que ejecutarlo una vez (tarda ~2 minutos):

```bash
python preprocess_dataset.py
```

Crea `predicted_labels.json` con las etiquetas que ha predicho nuestro sistema para
todo el dataset. El frontend lo carga en memoria al arrancar.

### 3. Lanzar el frontend web

```bash
python app/app.py
```

Despues abrir el navegador en **http://127.0.0.1:5001**:

- **Subir imagen** → te dice la forma y los colores predominantes.
- **Buscar por filtros** → muestra las imagenes del catalogo que cumplen la combinacion
  de forma y color seleccionada.

> Nota para macOS: la app usa el puerto 5001 porque el clasico 5000 esta ocupado por
> el AirPlay Receiver del sistema (Control Center).

### 4. Ejecutar los 4 analisis de eficiencia

```bash
python analysis.py
```

Tarda unos minutos. Los resultados se guardan en `informe/results.json` y se incluyen
en el informe.

### 5. Analisis metodologico de duplicados (extra)

```bash
python analyze_duplicates.py
```

Detecta las imagenes duplicadas binariamente entre train y test (~25% del test),
recalcula el accuracy del KNN sin esas duplicadas y guarda el resultado en
`informe/duplicates_analysis.json`. NO modifica el split oficial.

### 6. Regenerar las graficas del informe

```bash
python informe/generar_graficas.py
```

Lee los JSON de resultados y produce las 8 figuras en `informe/figures/` que se
insertan en `informe/informe.md`.

## Documento del informe

El informe completo (Introduccion, Desarrollo, Analisis de eficiencia, Conclusiones)
esta en `informe/informe.md`.

## Autor

Alejandro Marinas — Aprenentatge Automatic, La Salle Gracia (curso 2025-2026).
