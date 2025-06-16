import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# --- Script para Analizar Propiedades de Imágenes con Histogramas ---

# --- Setup de directorios ---
image_dir = "kaggle_flower_images"
plots_dir = "plots"
labels_file = os.path.join(image_dir, "flower_labels.csv")

# Asegurarse que el directorio de plots existe
os.makedirs(plots_dir, exist_ok=True)

# Carga de datos
try:
    labels_df = pd.read_csv(labels_file)
    image_files = [os.path.join(image_dir, f) for f in labels_df['file']]
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de etiquetas en {labels_file}")
    exit()

print("Analizando propiedades de las imágenes...")

# --- 1. Análisis de Tamaño (Dimensiones) de las Imágenes ---
print("\n--- 1. Analizando dimensiones de las imágenes ---")
image_shapes = [cv2.imread(f).shape for f in image_files if os.path.exists(f)]

shape_counts = Counter(image_shapes)

if len(shape_counts) == 1:
    print(f"Todas las imágenes tienen las mismas dimensiones: {list(shape_counts.keys())[0]}")
else:
    print("Se encontraron imágenes con diferentes dimensiones:")
    for shape, count in shape_counts.items():
        print(f"- Dimensiones {shape}: {count} imágenes")
    
    # Graficar la distribución de tamaños
    shapes_str = [str(s) for s in shape_counts.keys()]
    counts = list(shape_counts.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(shapes_str, counts, color='skyblue')
    plt.title('Distribución de Dimensiones de Imágenes')
    plt.xlabel('Dimensiones (Altura, Anchura, Canales)')
    plt.ylabel('Cantidad de Imágenes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    size_plot_path = os.path.join(plots_dir, 'image_dimensions_distribution.png')
    plt.savefig(size_plot_path)
    plt.show()
    print(f"Gráfico de dimensiones guardado en: {size_plot_path}")

# # --- 2. Histograma de Colores y Valores para una Imagen de Muestra ---
# print("\n--- 2. Generando histograma para una imagen de muestra ---")
# sample_image_path = image_files[0]
# sample_image = cv2.imread(sample_image_path)
# sample_image_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(10, 7))

# # Plot de la imagen de muestra
# plt.subplot(2, 1, 1)
# plt.imshow(sample_image_rgb)
# plt.title(f'Imagen de Muestra: {os.path.basename(sample_image_path)}')
# plt.axis('off')

# # Plot del histograma
# plt.subplot(2, 1, 2)
# colors = ('r', 'g', 'b')
# color_labels = ('Rojo', 'Verde', 'Azul')
# for i, color in enumerate(colors):
#     # La imagen en RGB tiene los canales en orden R, G, B. cv2.split en BGR los daría en orden B, G, R.
#     hist = cv2.calcHist([sample_image_rgb], [i], None, [256], [0, 256])
#     plt.plot(hist, color=color, label=f'Canal {color_labels[i]}')

# plt.title('Histograma de Distribución de Colores')
# plt.xlabel('Intensidad del Píxel')
# plt.ylabel('Cantidad de Píxeles')
# plt.xlim([0, 256])
# plt.legend()
# plt.grid(alpha=0.5)
# plt.tight_layout()

# sample_hist_path = os.path.join(plots_dir, 'sample_image_histogram.png')
# plt.savefig(sample_hist_path)
# plt.show()
# print(f"Histograma de muestra guardado en: {sample_hist_path}")


# --- 3. Histograma Global de Colores para todo el Dataset ---
print("\n--- 3. Calculando histograma global para todo el dataset ---")

# Inicializar acumuladores para los histogramas
r_hist_total = np.zeros(256)
g_hist_total = np.zeros(256)
b_hist_total = np.zeros(256)

for f in image_files:
    if os.path.exists(f):
        img = cv2.imread(f) # BGR
        b, g, r = cv2.split(img)
        r_hist_total += cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
        g_hist_total += cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        b_hist_total += cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()

# Graficar el histograma global
plt.figure(figsize=(10, 6))
plt.title('Histograma de Color Global para Todo el Dataset')
plt.xlabel('Intensidad del Píxel')
plt.ylabel('Cantidad Total de Píxeles (log scale)')
plt.plot(r_hist_total, color='red', label='Canal Rojo')
plt.plot(g_hist_total, color='green', label='Canal Verde')
plt.plot(b_hist_total, color='blue', label='Canal Azul')
plt.xlim([0, 256])
plt.yscale('log') # Usar escala logarítmica para ver mejor las distribuciones
plt.legend()
plt.grid(alpha=0.5)

global_hist_path = os.path.join(plots_dir, 'global_color_histogram.png')
plt.savefig(global_hist_path)
plt.show()
print(f"Histograma global guardado en: {global_hist_path}")

print("\nAnálisis completado. Los valores de los píxeles están en el rango estándar [0, 255].") 