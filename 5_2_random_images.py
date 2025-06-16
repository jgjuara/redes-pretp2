import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

# Establecer semilla para reproducibilidad
np.random.seed(42)

# --- 5.2 Generar imágenes random ---

# Directorio de imágenes
image_dir = "kaggle_flower_images"

# --- 1. Imagen mezclando los píxeles ---
print("Generando imagen con píxeles mezclados...")

# Cargar una imagen de ejemplo
sample_image_path = os.path.join(image_dir, "0001.png") # Usar la imagen 0001.png como base
original_image = cv2.imread(sample_image_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Aplanar la imagen a una lista de píxeles
pixels = original_image.reshape(-1, 3)
np.random.shuffle(pixels)

# Reconstruir la imagen con los píxeles mezclados
shuffled_image = pixels.reshape(original_image.shape)
shuffled_image_rgb = cv2.cvtColor(shuffled_image, cv2.COLOR_BGR2RGB)

# Graficar
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image_rgb)
plt.title("Imagen Original 0001.png")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(shuffled_image_rgb)
plt.title("Píxeles Mezclados")
plt.axis('off')

plt.suptitle("Generación de Imagen con Píxeles Mezclados Aleatoriamente", fontsize=16)
plt.savefig('plots/random_pixels_shuffled.png')
plt.show()


# --- 2. Imagen mezclando partes de diferentes imágenes ---
print("\nGenerando imagen mezclando partes de otras imágenes...")

# Cargar el archivo de etiquetas
labels_file = os.path.join(image_dir, "flower_labels.csv")
labels_df = pd.read_csv(labels_file)

# Filtrar para obtener 2 imágenes de la clase 0
class_0_files = labels_df[labels_df['label'] == 0]['file'].tolist()
# Filtrar para obtener 2 imágenes de la clase 7
class_7_files = labels_df[labels_df['label'] == 7]['file'].tolist()

# Seleccionar 2 imágenes de cada clase
# random.seed(42) # Asegurar que la muestra sea la misma cada vez
selected_files_0 = random.sample(class_0_files, 2)
selected_files_7 = random.sample(class_7_files, 2)

# Combinar las listas de archivos y crear las rutas completas
selected_files = selected_files_0 + selected_files_7
random_image_paths = [os.path.join(image_dir, f) for f in selected_files]

print(f"Imágenes seleccionadas: {selected_files}")

images = [cv2.imread(p) for p in random_image_paths]


height, width, _ = images[0].shape
part_h, part_w = height // 2, width // 2

# Crear una imagen nueva vacía
composite_image = np.zeros_like(images[0])

# Combinar las 4 partes
# Top-Left from image 1
composite_image[0:part_h, 0:part_w] = images[0][0:part_h, 0:part_w]
# Top-Right from image 2
composite_image[0:part_h, part_w:width] = images[1][0:part_h, part_w:width]
# Bottom-Left from image 3
composite_image[part_h:height, 0:part_w] = images[2][part_h:height, 0:part_w]
# Bottom-Right from image 4
composite_image[part_h:height, part_w:width] = images[3][part_h:height, part_w:width]

composite_image_rgb = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)

# Graficar las 4 imágenes originales y la imagen compuesta
plt.figure(figsize=(12, 10))
plt.suptitle("Generación de Imagen Compuesta", fontsize=16)

for i, img_path in enumerate(random_image_paths):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img_rgb)
    plt.title(f"Fuente {i+1}")
    plt.axis('off')

plt.subplot(2, 3, 5) # Poner la imagen compuesta en el centro
plt.imshow(composite_image_rgb)
plt.title("Imagen Compuesta")
plt.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/composite_image.png')
plt.show() 