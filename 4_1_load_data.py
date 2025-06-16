#%%

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# --- 4.1 Cargar el dataset y sus respectivas etiquetas ---

# Directorio de imágenes y archivo de etiquetas
image_dir = "kaggle_flower_images"
labels_file = os.path.join(image_dir, "flower_labels.csv")

# Cargar etiquetas
labels_df = pd.read_csv(labels_file)

# Cargar rutas de imágenes
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])

# Asegurarse de que el número de imágenes y etiquetas coincida
print(f"Número de imágenes encontradas: {len(image_files)}")
print(f"Número de etiquetas encontradas: {len(labels_df)}")

if len(image_files) == len(labels_df):
    print("El número de imágenes y etiquetas coincide.")
else:
    print("¡Advertencia! El número de imágenes y etiquetas no coincide.")

# Añadir rutas de archivo al DataFrame de etiquetas para facilitar el acceso
# Asumiendo que los archivos de imagen están nombrados '0001.png', '0002.png', etc.
# y que el 'file' en el CSV es '0001.png'
labels_df['filepath'] = labels_df['file'].apply(lambda x: os.path.join(image_dir, x))


# --- Mostrar una imagen de ejemplo y su etiqueta ---
if not labels_df.empty:
    sample_index = 0
    sample_image_path = labels_df.loc[sample_index, 'filepath']
    sample_label = labels_df.loc[sample_index, 'label']

    print(f"\nMostrando imagen de ejemplo: {os.path.basename(sample_image_path)}")
    print(f"Etiqueta: {sample_label}")

    # Cargar y mostrar la imagen
    image = cv2.imread(sample_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convertir de BGR a RGB para Matplotlib

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.title(f"Label: {sample_label}")
    plt.axis('off')
    plt.show()
else:
    print("\nNo se pudieron cargar las etiquetas o no hay imágenes.")

# --- Verificación de comparabilidad (tamaño) ---
# Se asume que todas las imágenes tienen el mismo tamaño como se indica en el TP (128x128)
# Verifiquemos el tamaño de la primera imagen
first_image = cv2.imread(image_files[0])
print(f"\nDimensiones de la primera imagen: {first_image.shape}")
print("Se asume que todas las imágenes son 128x128x3 y están en el mismo espacio de color (RGB).") 