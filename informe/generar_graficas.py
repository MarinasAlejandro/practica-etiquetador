"""Genera las graficas que acompanan al informe.

Carga los resultados ya calculados (informe/results.json,
informe/duplicates_analysis.json) y produce un set de figuras en
informe/figures/ que se insertan en informe.md.

Estilo deliberadamente sobrio (whitegrid + paleta tab10) para que las
graficas se parezcan a las de los slides de teoria.
"""

import json
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Permitimos lanzar este script desde la raiz o desde informe/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import Kmeans  # noqa: E402
from KNN import KNN  # noqa: E402
from utils_data import read_dataset  # noqa: E402
import utils  # noqa: E402


# -----------------------------------------------------------------------------
# Configuracion global de estilo
# -----------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFORME_DIR = os.path.join(ROOT, 'informe')
FIG_DIR = os.path.join(INFORME_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 120


def guardar(nombre):
    path = os.path.join(FIG_DIR, nombre)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  guardado: figures/{nombre}")


# -----------------------------------------------------------------------------
# 1. Distribucion de clases (train vs test)
# -----------------------------------------------------------------------------

def fig_distribucion_clases(train_class, test_class):
    print("Fig 1: distribucion de clases")
    clases = sorted(set(train_class.tolist()) | set(test_class.tolist()))
    train_count = [int(np.sum(train_class == c)) for c in clases]
    test_count = [int(np.sum(test_class == c)) for c in clases]

    y = np.arange(len(clases))
    h = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(y - h/2, train_count, h, label=f'train (n={len(train_class)})', color='#1f77b4')
    ax.barh(y + h/2, test_count, h, label=f'test  (n={len(test_class)})', color='#ff7f0e')
    ax.set_yticks(y)
    ax.set_yticklabels(clases)
    ax.invert_yaxis()
    ax.set_xlabel('Numero de imagenes')
    ax.set_title('Distribucion de clases por split')
    ax.legend(loc='lower right')
    guardar('fig1_distribucion_clases.png')


# -----------------------------------------------------------------------------
# 2. Accuracy del KNN por k (analisis 2)
# -----------------------------------------------------------------------------

def fig_knn_accuracy_k(results):
    print("Fig 2: accuracy KNN por k")
    datos = results['analisis_2_k_knn']
    ks = sorted(int(k) for k in datos.keys())
    accs = [datos[str(k)]['accuracy'] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, accs, marker='o', linewidth=2, color='#1f77b4')
    for k, a in zip(ks, accs):
        ax.annotate(f'{a:.2f}%', (k, a), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9)
    ax.set_xlabel('k (numero de vecinos)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy del KNN segun k')
    ax.set_xticks(ks)
    ax.set_ylim(min(accs) - 1.5, max(accs) + 1.5)
    guardar('fig2_knn_accuracy_k.png')


# -----------------------------------------------------------------------------
# 3. Curva del codo de WCD (find_bestK) sobre una imagen ejemplo
# -----------------------------------------------------------------------------

def fig_codo_wcd(img_path):
    print("Fig 3: curva del codo (WCD por K)")
    img = np.array(Image.open(img_path).convert('RGB').resize((60, 80)))
    ks = list(range(2, 9))
    wcds = []
    for k in ks:
        km = Kmeans.KMeans(img, K=k)
        km.fit()
        wcds.append(km.withinClassDistance())

    decs = [None] + [100 - 100 * (wcds[i] / wcds[i-1]) for i in range(1, len(wcds))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(ks, wcds, marker='o', linewidth=2, color='#2ca02c')
    ax1.set_xlabel('K (numero de clusters)')
    ax1.set_ylabel('WCD (intra-cluster)')
    ax1.set_title('Curva del codo: WCD por K')

    ax2.bar(ks[1:], decs[1:], color='#ff7f0e', edgecolor='black')
    ax2.axhline(20, linestyle='--', color='red', label='umbral 20% (PDF)')
    ax2.set_xlabel('K (numero de clusters)')
    ax2.set_ylabel('Decremento de WCD (%)')
    ax2.set_title('Criterio del 20%: cuando paramos de anadir clusters')
    ax2.legend()

    guardar('fig3_codo_wcd.png')


# -----------------------------------------------------------------------------
# 4. K elegida por umbral (analisis 3)
# -----------------------------------------------------------------------------

def fig_umbrales_bestK(results):
    print("Fig 4: K elegida por umbral en find_bestK")
    datos = results['analisis_3_criterio_bestK']
    umbrales = sorted(int(u) for u in datos.keys())
    medias = [datos[str(u)]['K_media'] for u in umbrales]
    mins = [datos[str(u)]['K_min'] for u in umbrales]
    maxs = [datos[str(u)]['K_max'] for u in umbrales]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(umbrales, medias,
                yerr=[np.array(medias) - np.array(mins), np.array(maxs) - np.array(medias)],
                fmt='o-', capsize=5, linewidth=2, color='#9467bd',
                label='K media (con min/max)')
    ax.axvline(20, linestyle='--', color='red', alpha=0.7, label='umbral del PDF (20%)')
    ax.set_xlabel('Umbral de decremento (%)')
    ax.set_ylabel('K elegida')
    ax.set_title('Criterio find_bestK: K media por umbral (N=30 imagenes)')
    ax.legend()
    guardar('fig4_umbrales_bestK.png')


# -----------------------------------------------------------------------------
# 5. Iteraciones hasta convergencia (analisis 1)
# -----------------------------------------------------------------------------

def fig_inicializacion(results):
    print("Fig 5: comparativa inicializaciones first vs random")
    datos = results['analisis_1_inicializacion']
    estrategias = ['first', 'random']
    iter_media = [datos[e]['iter_media'] for e in estrategias]
    wcd_media = [datos[e]['wcd_media'] for e in estrategias]
    t_media = [datos[e]['t_medio_ms'] for e in estrategias]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    color_map = {'first': '#1f77b4', 'random': '#ff7f0e'}
    colores = [color_map[e] for e in estrategias]

    axes[0].bar(estrategias, iter_media, color=colores, edgecolor='black')
    axes[0].set_title('Iteraciones medias')
    axes[0].set_ylabel('Iteraciones')

    axes[1].bar(estrategias, wcd_media, color=colores, edgecolor='black')
    axes[1].set_title('WCD media final')
    axes[1].set_ylabel('WCD')

    axes[2].bar(estrategias, t_media, color=colores, edgecolor='black')
    axes[2].set_title('Tiempo medio (ms)')
    axes[2].set_ylabel('ms por imagen')

    for ax, valores in zip(axes, [iter_media, wcd_media, t_media]):
        for i, v in enumerate(valores):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Inicializacion first vs random (K=4, N=50 imagenes)',
                 fontweight='bold')
    guardar('fig5_inicializacion.png')


# -----------------------------------------------------------------------------
# 6. Vista visual del K-Means sobre una prenda (RGB + cuantizacion)
# -----------------------------------------------------------------------------

def fig_kmeans_visual(img_path, K=4):
    print("Fig 6: K-Means visual sobre una prenda")
    img = np.array(Image.open(img_path).convert('RGB').resize((60, 80)))
    km = Kmeans.KMeans(img, K=K)
    km.fit()
    cuantizada = km.centroids[km.labels].reshape(img.shape).astype(np.uint8)
    nombres = Kmeans.get_colors(km.centroids)

    fig = plt.figure(figsize=(11, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img)
    ax1.set_title('Imagen original')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(cuantizada)
    ax2.set_title(f'Cuantizada con K={K} colores')
    ax2.axis('off')

    # Paleta de centroides en orden
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    Xs = km.X[::20]
    labels = km.labels[::20]
    for k in range(km.K):
        pts = Xs[labels == k]
        if len(pts) == 0:
            continue
        ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4,
                    c=[km.centroids[k] / 255.0])
    ax3.set_xlabel('R')
    ax3.set_ylabel('G')
    ax3.set_zlabel('B')
    ax3.set_title(f'Pixeles en RGB ({" + ".join(nombres)})')

    fig.suptitle(
        'K-Means analiza TODOS los pixeles RGB de la imagen, incluyendo el fondo.\n'
        'Por eso suele aparecer Blanco/Gris cuando el fondo es uniforme.',
        fontsize=10
    )
    guardar('fig6_kmeans_visual.png')


# -----------------------------------------------------------------------------
# 7. Analisis de duplicados (accuracy oficial vs limpio)
# -----------------------------------------------------------------------------

def fig_duplicados(dup_results):
    print("Fig 7: accuracy con/sin duplicados")
    labels = ['Test entero\n(oficial)', 'Sin duplicados\n(limpio)', 'Solo duplicados']
    valores = [
        dup_results['accuracy_oficial_k3'],
        dup_results['accuracy_sin_duplicados_k3'],
        dup_results['accuracy_solo_duplicados_k3'],
    ]
    colores = ['#1f77b4', '#2ca02c', '#d62728']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.bar(labels, valores, color=colores, edgecolor='black')
    for i, v in enumerate(valores):
        ax1.text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy del KNN (k=3) segun subset del test')
    ax1.set_ylim(80, 100)

    # Pie con la fraccion duplicada
    n_dup = dup_results['n_duplicados']
    n_total = dup_results['n_test_archivos']
    sizes = [n_dup, n_total - n_dup]
    ax2.pie(sizes,
            labels=[f'duplicadas ({n_dup})', f'unicas ({n_total - n_dup})'],
            colors=['#d62728', '#bbbbbb'],
            autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Composicion del test ({n_total} imagenes)')

    fig.suptitle('Imagenes duplicadas entre train y test',
                 fontweight='bold')
    guardar('fig7_duplicados.png')


# -----------------------------------------------------------------------------
# 8. Matriz de confusion del KNN (k=3)
# -----------------------------------------------------------------------------

def fig_matriz_confusion(train_imgs, train_class, test_imgs, test_class, k=3):
    print(f"Fig 8: matriz de confusion KNN (k={k})")
    clases = sorted(set(train_class.tolist()))
    idx = {c: i for i, c in enumerate(clases)}

    knn = KNN(train_imgs, train_class)
    pred = knn.predict(test_imgs, k=k)

    n = len(clases)
    mat = np.zeros((n, n), dtype=int)
    for real, p in zip(test_class, pred):
        mat[idx[real], idx[p]] += 1

    # Normalizamos por fila para verlo en porcentaje
    mat_norm = mat / mat.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat_norm, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(clases, rotation=45, ha='right')
    ax.set_yticklabels(clases)
    ax.set_xlabel('Prediccion')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de confusion KNN (k={k}) - % por fila')

    for i in range(n):
        for j in range(n):
            color = 'white' if mat_norm[i, j] > 50 else 'black'
            ax.text(j, i, f'{mat[i, j]}', ha='center', va='center',
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label='% por fila')
    guardar('fig8_matriz_confusion.png')


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("\n=== Generando graficas del informe ===\n")

    with open(os.path.join(INFORME_DIR, 'results.json')) as f:
        results = json.load(f)
    with open(os.path.join(INFORME_DIR, 'duplicates_analysis.json')) as f:
        dup_results = json.load(f)

    train_imgs, train_class, _, test_imgs, test_class, _ = read_dataset(
        root_folder=os.path.join(ROOT, 'images') + '/',
        gt_json=os.path.join(ROOT, 'images', 'gt.json'),
    )

    # Imagen ejemplo para visualizar K-Means
    img_ejemplo = os.path.join(ROOT, 'images', 'train', '10004.jpg')

    fig_distribucion_clases(train_class, test_class)
    fig_knn_accuracy_k(results)
    fig_codo_wcd(img_ejemplo)
    fig_umbrales_bestK(results)
    fig_inicializacion(results)
    fig_kmeans_visual(img_ejemplo, K=4)
    fig_duplicados(dup_results)
    fig_matriz_confusion(train_imgs, train_class, test_imgs, test_class, k=3)

    print(f"\nTodas las graficas en {FIG_DIR}")


if __name__ == '__main__':
    main()
