import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 6.1 Analizar las distribuciones de valores de píxeles por cada especie ---

# Directorio de imágenes y archivo de etiquetas
image_dir = "kaggle_flower_images"
labels_file = os.path.join(image_dir, "flower_labels.csv")

# Carga de datos
try:
    labels_df = pd.read_csv(labels_file)
    labels_df['filepath'] = labels_df['file'].apply(lambda x: os.path.join(image_dir, x))
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de etiquetas en {labels_file}")
    exit()

# --- Análisis de la distribución de píxeles ---
species = labels_df['label'].unique()
colors = ('blue', 'green', 'red') # OpenCV carga en orden BGR
color_labels = ('Azul (B)', 'Verde (G)', 'Rojo (R)')

# Crear una figura para las distribuciones de todas las especies
plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(len(species), 1, figsize=(12, 2 * len(species)), sharex=True, sharey=True)
fig.suptitle('Distribución de Intensidad de Píxeles por Especie y Canal de Color', fontsize=20)

print("Calculando y graficando las distribuciones de píxeles por especie...")

for i, s in enumerate(species):
    print(f"Procesando {s}/{len(species)}")
    ax = axes[i]
    species_paths = labels_df[labels_df['label'] == s]['filepath'].tolist()
    
    # Acumular los valores de los píxeles para todos los canales
    pixel_values = { 'blue': [], 'green': [], 'red': [] }

    for path in species_paths:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                # Separar los canales
                b, g, r = cv2.split(img)
                pixel_values['blue'].extend(b.flatten())
                pixel_values['green'].extend(g.flatten())
                pixel_values['red'].extend(r.flatten())

    # Graficar la distribución para cada canal de color usando KDE
    for color, label in zip(colors, color_labels):
        sns.kdeplot(pixel_values[color], ax=ax, color=color, label=label, fill=True, alpha=0.2)
    
    ax.set_title(f'Especie: {s}')
    ax.set_xlabel('Intensidad del Píxel (0-255)')
    ax.set_ylabel('Densidad')
    ax.legend()
    ax.set_xlim(0, 255)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(f'plots/histograma_colores_especies.png')
plt.show()

print("""
--- ¿Se puede distinguir una especie en algún rango de color? ---
Al observar los gráficos de densidad, podemos hacer las siguientes observaciones:

- **Picos de Intensidad:** Algunas especies pueden mostrar picos de intensidad característicos en ciertos canales de color. Por ejemplo, una flor predominantemente roja (como una rosa) tendrá una distribución de píxeles del canal 'Rojo' sesgada hacia valores altos (cercanos a 255). Una flor amarilla tendrá picos altos tanto en el canal Rojo como en el Verde.

- **Separación de Curvas:** Si para una especie, la curva de un canal de color está claramente separada de las curvas de otras especies, ese rango de color podría ser un buen 'feature' para distinguirla. Por ejemplo, si solo los girasoles (sunflower) tienen una alta densidad de píxeles con valores R y G > 200, este hecho puede ser usado para su identificación.

- **Dominancia del Negro/Blanco:** Todas las especies muestran un pico en valores bajos (cercanos a 0), que corresponde al fondo negro de las imágenes. De manera similar, puede haber picos en valores altos (cercanos a 255) si hay áreas blancas o muy brillantes.

**Conclusión:** Sí, es posible distinguir algunas especies si sus distribuciones de color son lo suficientemente únicas. Sin embargo, para flores con colores similares, este método por sí solo podría no ser suficiente y se necesitarían características más complejas (como textura o forma).
-----------------------------------------------------------------
""") 