import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 6.3 Analizar las distribuciones de valores de píxeles en imágenes en escala de grises ---
# Este script analiza la distribución de píxeles en escala de grises
# para cada especie de flor.

# Directorio de imágenes y archivo de etiquetas
image_dir = "kaggle_flower_images"
labels_file = os.path.join(image_dir, "flower_labels.csv")
output_dir = "plots"

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Carga de datos
try:
    labels_df = pd.read_csv(labels_file)
    labels_df['filepath'] = labels_df['file'].apply(lambda x: os.path.join(image_dir, x))
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de etiquetas en {labels_file}")
    exit()

# --- Análisis de la distribución de píxeles en imágenes en escala de grises ---
species = labels_df['label'].unique()

# Crear una figura para las distribuciones de todas las especies
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(len(species), 1, figsize=(12, 2 * len(species)), sharex=True, sharey=True)
fig.suptitle('Distribución de Intensidad de Píxeles en Escala de Grises por Especie', fontsize=20)

print("Calculando y graficando las distribuciones de píxeles en escala de grises por especie...")

for i, s in enumerate(species):
    print(f"  Procesando especie {i+1}/{len(species)}: {s}")
    ax = axes[i]
    species_paths = labels_df[labels_df['label'] == s]['filepath'].tolist()
    
    # Acumular los valores de los píxeles para todas las imágenes en escala de grises
    pixel_values = []

    for path in species_paths:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                # Convertir a escala de grises
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                pixel_values.extend(gray.flatten())

    # Graficar la distribución para los píxeles en escala de grises usando KDE
    sns.kdeplot(pixel_values, ax=ax, fill=True, alpha=0.2, color='black')
    
    ax.set_title(f'Especie: {s}')
    ax.set_xlim(0, 255)

# Añadir etiquetas centrales para los ejes X e Y
fig.supxlabel('Intensidad del Píxel (0-255)', fontsize=14)
fig.supylabel('Densidad', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Guardar el gráfico
output_path = os.path.join(output_dir, "6_3_pixel_distribution_grayscale.png")
plt.savefig(output_path)
print(f"Gráfico guardado en: {output_path}")

plt.show()

print("""
--- ¿Se puede distinguir una especie por su distribución de píxeles en escala de grises? ---
Al observar los gráficos de densidad de las imágenes en escala de grises, podemos analizar:

- **Distribución de Luminosidad:** Cada gráfico muestra la distribución de la intensidad de los píxeles (brillo) para una especie. Los picos en la distribución indican las intensidades más comunes.

- **Pico del Fondo:** De manera similar a los análisis de color, se observa un pico fuerte cerca de 0 en todas las especies, que corresponde al fondo negro de las imágenes.

- **Características de la Especie:** La forma de la distribución más allá del pico del fondo puede revelar características de la especie. Por ejemplo, flores con colores naturalmente claros (como margaritas o algunas rosas) mostrarán una mayor densidad de píxeles en la zona de intensidad media-alta. Flores oscuras (como algunos tulipanes) tendrán su distribución más concentrada en la zona de intensidad baja-media.

**Conclusión:** La distribución de intensidad en escala de grises puede ser un rasgo distintivo. Aunque se pierde la información de color, la luminosidad general de una flor puede ayudar a diferenciarla de otras. Especies con colores muy diferentes que tienen una luminosidad similar serían difíciles de distinguir con este método.
-----------------------------------------------------------------
""") 