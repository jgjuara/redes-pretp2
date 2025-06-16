import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 5.4 Calcular imagen promedio global y por especie ---

# Directorio de imágenes y archivo de etiquetas
image_dir = "kaggle_flower_images"
labels_file = os.path.join(image_dir, "flower_labels.csv")

# --- Funciones de ayuda ---
def load_images_from_paths(paths, color_mode=cv2.IMREAD_COLOR):
    """Carga imágenes desde una lista de rutas."""
    images = []
    for path in paths:
        if os.path.exists(path):
            img = cv2.imread(path, color_mode)
            # Asegurarse que la imagen se cargó correctamente
            if img is not None:
                images.append(img)
    return images

def calculate_average_image(images):
    """Calcula la imagen promedio de una lista de imágenes."""
    if not images:
        return None
    # Convertir a float para promediar, luego volver a uint8 para mostrar
    avg_image = np.mean(images, axis=0).astype(np.uint8)
    return avg_image

def process_and_display_averages(labels_df, color_mode=cv2.IMREAD_COLOR, image_type="Color"):
    """Carga, procesa y muestra las imágenes promedio."""
    
    # Cargar todas las rutas de imágenes
    all_image_paths = labels_df['filepath'].tolist()
    all_images = load_images_from_paths(all_image_paths, color_mode=color_mode)

    if not all_images:
        print(f"No se pudieron cargar imágenes para el tipo: {image_type}")
        return

    # 1. Calcular promedio global
    global_avg_image = calculate_average_image(all_images)

    if global_avg_image is not None:
        plt.figure(figsize=(5, 5))
        if image_type == "Color":
            plt.imshow(cv2.cvtColor(global_avg_image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(global_avg_image, cmap='gray')
        plt.title(f"Imagen Promedio Global ({image_type})")
        plt.axis('off')
        plt.show()

    # 2. Calcular promedio por especie
    species = labels_df['label'].unique()
    n_species = len(species)
    
    plt.figure(figsize=(15, n_species * 3))
    plt.suptitle(f"Imágenes Promedio por Especie ({image_type})", fontsize=16)

    for i, s in enumerate(species):
        species_paths = labels_df[labels_df['label'] == s]['filepath'].tolist()
        species_images = load_images_from_paths(species_paths, color_mode=color_mode)
        species_avg_image = calculate_average_image(species_images)
        
        if species_avg_image is not None:
            plt.subplot(n_species // 2 + n_species % 2, 2, i + 1)
            if image_type == "Color":
                plt.imshow(cv2.cvtColor(species_avg_image, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(species_avg_image, cmap='gray')
            plt.title(f"Promedio: {s}")
            plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    print(f"\n--- Análisis de Distinguibilidad ({image_type}) ---")
    if image_type == "Color":
        print("Las imágenes promedio a color pueden mostrar diferencias sutiles en la tonalidad dominante de cada especie.")
        print("Por ejemplo, algunas flores pueden tener un promedio más rojizo o amarillento, lo que las hace distinguibles.")
    else:
        print("En blanco y negro (o binarizadas), se pierde la información de color. La distinción depende de la forma y estructura.")
        print("Los promedios tienden a parecerse más entre sí, mostrando una 'forma de flor' genérica. Puede ser difícil distinguir especies, a menos que una tenga una forma muy característica (ej. girasoles vs. rosas).")
    print("-" * 50)


# --- Carga de datos principal ---
try:
    labels_df = pd.read_csv(labels_file)
    labels_df['filepath'] = labels_df['file'].apply(lambda x: os.path.join(image_dir, x))
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de etiquetas en {labels_file}")
    exit()

# --- Procesar para imágenes a color ---
print("Calculando promedios para imágenes a COLOR...")
process_and_display_averages(labels_df, color_mode=cv2.IMREAD_COLOR, image_type="Color")


# --- Procesar para imágenes en blanco y negro (binarizadas) ---
def load_and_binarize_images(paths, threshold_value=127):
    """Carga imágenes, las convierte a escala de grises y las binariza."""
    images = []
    for path in paths:
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
                images.append(binary_img)
    return images

def process_and_display_binary_averages(labels_df):
    """Carga, binariza y muestra los promedios."""
    all_image_paths = labels_df['filepath'].tolist()
    all_images_binary = load_and_binarize_images(all_image_paths)

    if not all_images_binary:
        print("No se pudieron cargar y binarizar imágenes.")
        return

    # Promedio global binario
    global_avg_binary = calculate_average_image(all_images_binary)
    if global_avg_binary is not None:
        plt.figure(figsize=(5, 5))
        plt.imshow(global_avg_binary, cmap='gray')
        plt.title("Imagen Promedio Global (Binarizada)")
        plt.axis('off')
        plt.show()

    # Promedio por especie binario
    species = labels_df['label'].unique()
    n_species = len(species)
    
    plt.figure(figsize=(15, n_species * 3))
    plt.suptitle("Imágenes Promedio por Especie (Binarizada)", fontsize=16)

    for i, s in enumerate(species):
        species_paths = labels_df[labels_df['label'] == s]['filepath'].tolist()
        species_images_binary = load_and_binarize_images(species_paths)
        species_avg_binary = calculate_average_image(species_images_binary)
        
        if species_avg_binary is not None:
            plt.subplot(n_species // 2 + n_species % 2, 2, i + 1)
            plt.imshow(species_avg_binary, cmap='gray')
            plt.title(f"Promedio Binarizado: {s}")
            plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

# Esta sección se deja comentada porque la binarización se hace dentro de la función
# print("\nCalculando promedios para imágenes en BLANCO Y NEGRO (Binarizadas)...")
# process_and_display_binary_averages(labels_df)

# La pregunta es "¿Cómo quedan los promedios si consideran las imágenes en blanco y negro?"
# El "promedio de binarizadas" es una interpretación. Otra es "promedio de grises".
# Mostraremos el promedio de las imágenes en escala de grises, que es más informativo.
print("\nCalculando promedios para imágenes en ESCALA DE GRISES...")
process_and_display_averages(labels_df, color_mode=cv2.IMREAD_GRAYSCALE, image_type="Grises") 