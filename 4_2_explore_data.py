import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# --- 4.2 Guardar imágenes de muestra y crear un grid de especies ---

# --- Configuración ---
image_dir = "kaggle_flower_images"
labels_file = os.path.join(image_dir, "flower_labels.csv")
output_dir = "plots"

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# --- Carga de datos ---
try:
    labels_df = pd.read_csv(labels_file)
    # Añadir rutas de archivo al DataFrame
    labels_df['filepath'] = labels_df['file'].apply(lambda x: os.path.join(image_dir, x))
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de etiquetas en {labels_file}")
    exit()

# --- Obtener especies y preparar la figura ---
species = labels_df['label'].unique()
print(f"Especies encontradas ({len(species)}): {', '.join(species.astype(str))}")

# Crear una figura con una cuadrícula de 3x4 para las 10 especies
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle("Muestra de una imagen por cada especie de flor", fontsize=20)
axes = axes.flatten() # Aplanar el array de ejes para iterar fácilmente

# --- Procesar cada especie ---
for i, s in enumerate(species):
    # Tomar una muestra aleatoria de una imagen para la especie actual
    sample_df = labels_df[labels_df['label'] == s].sample(n=1, random_state=42)
    
    if not sample_df.empty:
        image_path = sample_df.iloc[0]['filepath']
        
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            
            # 1. Guardar una imagen individual de la especie en /plots
            individual_save_path = os.path.join(output_dir, f"{s}.png")
            cv2.imwrite(individual_save_path, image)
            print(f"Guardada imagen de muestra para '{s}' en: {individual_save_path}")

            # 2. Añadir la imagen a la cuadrícula
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax = axes[i]
            ax.imshow(image_rgb)
            ax.set_title(s, fontsize=14)
            ax.axis('off')
        else:
            print(f"Advertencia: No se encontró el archivo de imagen {image_path}")
            axes[i].axis('off') # Ocultar el subplot si la imagen no existe
    else:
        axes[i].axis('off') # Ocultar si no hay imágenes para la especie

# Ocultar los ejes no utilizados (la cuadrícula es 3x4=12, pero hay 10 especies)
for i in range(len(species), len(axes)):
    axes[i].axis('off')

# --- Guardar la figura de la cuadrícula ---
plt.tight_layout(rect=[0, 0, 1, 0.96])
grid_save_path = os.path.join(output_dir, "flower_species_grid.png")
plt.savefig(grid_save_path)
print(f"\nCuadrícula de imágenes guardada en: {grid_save_path}") 