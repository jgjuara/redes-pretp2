import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


# 2. Realizar PCA
print("Realizando PCA para reducir a 2 componentes...")
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Crear un DataFrame con los resultados
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['label'] = labels

print(f"Varianza explicada por cada componente: {pca.explained_variance_ratio_}")
print(f"Varianza total explicada por 2 componentes: {np.sum(pca.explained_variance_ratio_):.2f}")


# 3. Graficar los resultados
print("Graficando los resultados de PCA...")
plt.figure(figsize=(14, 10))
sns.scatterplot(
    x="PC1", y="PC2",
    hue="label",
    palette=sns.color_palette("hsv", n_colors=len(labels_df['label'].unique())),
    data=pca_df,
    legend="full",
    alpha=0.8
)
plt.title('Análisis de Componentes Principales (PCA) de las Imágenes de Flores', fontsize=16)
plt.xlabel('Primer Componente Principal (PC1)')
plt.ylabel('Segundo Componente Principal (PC2)')
plt.legend(title='Especie')
plt.grid()
plt.show()

print("""
--- ¿Se pueden identificar las especies en esta representación? ---
El gráfico de PCA nos muestra una 'sombra' 2D de nuestros datos de muy alta dimensionalidad.

- **Agrupamientos (Clusters):** Si las especies son distintas en términos de sus características de píxeles (color, textura, forma), esperaríamos ver agrupamientos de puntos del mismo color en el gráfico. Si los puntos de una especie forman un grupo denso y separado de los demás, significa que PCA ha logrado capturar las características que hacen única a esa especie.

- **Superposición (Overlap):** Si los puntos de diferentes colores (especies) están muy mezclados, significa que, en las dos dimensiones principales de variación, estas especies son muy similares. Esto no implica que no se puedan separar, sino que la separación podría requerir más componentes principales o un método de reducción de dimensionalidad no lineal (como t-SNE o UMAP).

**Conclusión:** La visualización de PCA es una herramienta exploratoria poderosa. Generalmente, se pueden identificar algunas agrupaciones que corresponden a ciertas especies, especialmente aquellas con colores o formas muy distintivas (ej. girasoles amarillos vs. rosas rojas). Sin embargo, es común ver una superposición considerable entre especies que son visualmente parecidas. La baja varianza total explicada (si es el caso) también indica que las dos primeras componentes no capturan toda la complejidad de los datos.
-----------------------------------------------------------------
""") 