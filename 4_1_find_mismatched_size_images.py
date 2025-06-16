import os
import cv2
import pandas as pd

# --- Script para encontrar y redimensionar imágenes con dimensiones inesperadas ---

# --- Configuración ---
image_dir = "kaggle_flower_images"
labels_file = os.path.join(image_dir, "flower_labels.csv")
expected_shape = (128, 128, 3)
target_size = (expected_shape[1], expected_shape[0]) # (width, height)

# --- Carga de datos ---
try:
    labels_df = pd.read_csv(labels_file)
    # Crear la ruta completa para cada archivo de imagen
    image_files = [(row['file'], os.path.join(image_dir, row['file'])) for index, row in labels_df.iterrows()]
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de etiquetas en {labels_file}")
    exit()

print(f"Verificando {len(image_files)} imágenes. Buscando y corrigiendo dimensiones distintas de {expected_shape}...")

# --- Búsqueda y corrección de imágenes con dimensiones incorrectas ---
resized_images = []

for filename, filepath in image_files:
    if os.path.exists(filepath):
        image = cv2.imread(filepath)
        if image is not None:
            if image.shape != expected_shape:
                original_shape = image.shape
                
                # Redimensionar la imagen
                print(f"Redimensionando '{filename}' de {original_shape} a {target_size}...")
                resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                
                # Sobrescribir el archivo original
                cv2.imwrite(filepath, resized_image)
                
                resized_images.append({
                    "filename": filename,
                    "filepath": filepath,
                    "original_shape": original_shape,
                    "new_shape": resized_image.shape
                })
        else:
            print(f"Advertencia: No se pudo leer la imagen en {filepath}")
    else:
        print(f"Advertencia: No se encontró el archivo de imagen {filepath}")

# --- Reporte de resultados ---
if not resized_images:
    print(f"\n¡Éxito! Todas las imágenes ya tienen las dimensiones esperadas de {expected_shape}.")
else:
    print(f"\nSe redimensionaron {len(resized_images)} imágenes con dimensiones distintas a las esperadas:")
    print("-" * 50)
    for img_info in resized_images:
        print(f"Archivo: {img_info['filename']}")
        print(f"  - Dimensiones originales: {img_info['original_shape']}")
        print(f"  - Nuevas dimensiones: {img_info['new_shape']}")
    print("-" * 50)
    print("Todas las imágenes en el dataset han sido verificadas y/o ajustadas a 128x128.") 