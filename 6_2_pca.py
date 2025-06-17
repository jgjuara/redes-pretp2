import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# --- 6.2 Inspección de Componentes Principales (PCA) ---

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

# 1. Cargar y aplanar todas las imágenes
print("Cargando y preparando las imágenes para PCA...")
all_images_flattened = []
labels = []

for index, row in labels_df.iterrows():
    path = row['filepath']
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR) # Cargar a color
        if img is not None:
            # Aplanar la imagen (128x128x3 -> 49152) y añadirla a la lista
            all_images_flattened.append(img.flatten())
            labels.append(row['label'])

# Convertir la lista a un array de NumPy
X = np.array(all_images_flattened)
print(f"Dimensiones de la matriz de datos (Imágenes x Features): {X.shape}")

# Es buena práctica escalar los datos antes de PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- Análisis de Varianza Explicada ---
print("\nRealizando PCA para analizar la varianza explicada...")
# No especificamos n_components para calcularlos todos y ver la curva de varianza
pca_full = PCA().fit(X_scaled)

# Graficar la varianza explicada acumulada
print("Graficando la curva de varianza explicada acumulada...")
plt.figure(figsize=(10, 6))
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Varianza Explicada Acumulada vs. Número de Componentes')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.grid(True)
# Encontrar el número de componentes para explicar el 95% de la varianza
try:
    n_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
    plt.axhline(y=0.95, color='r', linestyle='-', label=f'95% Varianza ({n_components_95} componentes)')
    print(f"Número de componentes necesarios para capturar el 95% de la varianza: {n_components_95}")
except IndexError:
    print("No se alcanzó el 95% de la varianza.")

# Añadir línea vertical para n_components = 2
variance_at_2 = cumulative_variance[1] # El índice 1 corresponde a 2 componentes
plt.axvline(x=2, color='g', linestyle='--', label=f'Varianza con 2 componentes ({variance_at_2:.2f})')

plt.legend(loc='best')
plt.savefig("plots/pca_variance.png")
plt.show()
