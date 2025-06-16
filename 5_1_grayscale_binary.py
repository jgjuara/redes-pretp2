import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- 5.1 Convertir a escala de grises y binarizar ---

# Directorio de imágenes
image_dir = "kaggle_flower_images"
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])

# Seleccionar una imagen de ejemplo
sample_image_path = image_files[0]

# Cargar la imagen a color
image_color = cv2.imread(sample_image_path)
image_color_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# 1. Convertir la imagen a escala de grises y graficarla
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
# Crear directorio plots si no existe

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_color_rgb)
plt.title("Imagen Original (Color)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_gray, cmap='gray')
plt.title("Imagen en Escala de Grises")
plt.axis('off')

plt.suptitle("Conversión a Escala de Grises", fontsize=16)

# Guardar figura
plt.savefig('plots/grayscale_conversion.png')
plt.show()

# 2. Explicación de la binarización
print("""
--- Explicación de la Binarización ---
Para convertir una imagen en escala de grises a una imagen binaria (blanco y negro), 
necesitamos definir un 'umbral' (threshold). Cada píxel de la imagen en escala de grises 
tiene un valor de intensidad (normalmente entre 0 y 255).

El proceso de binarización funciona de la siguiente manera:
1. Se elige un valor de umbral (por ejemplo, 127).
2. Se recorre cada píxel de la imagen en escala de grises.
3. Si el valor de intensidad del píxel es MAYOR que el umbral, se le asigna un valor máximo (ej. 255, que representa el blanco).
4. Si el valor de intensidad del píxel es MENOR O IGUAL que el umbral, se le asigna un valor mínimo (ej. 0, que representa el negro).

El resultado es una imagen donde cada píxel es o blanco o negro, sin tonos intermedios.
La función `cv2.threshold` de OpenCV realiza esta operación de manera eficiente.
-----------------------------------------
""")

# 3. Aplicar binarización con 4 umbrales diferentes y graficar resultados
thresholds = np.linspace(0, 255, num=6, dtype=int)[1:-1] # 4 puntos intermedios: [51, 102, 153, 204]

for thresh in thresholds:
    # Aplicar el umbral
    ret, image_binary = cv2.threshold(image_gray, thresh, 255, cv2.THRESH_BINARY)
    
    # Crear la figura con dos subplots (imagen e histograma)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Subplot 1: Imagen Binarizada ---
    axes[0].imshow(image_binary, cmap='gray')
    axes[0].set_title(f"Imagen Binarizada (Umbral = {thresh})")
    axes[0].axis('off')
    
    # --- Subplot 2: Histograma ---
    axes[1].hist(image_gray.ravel(), 256, [0, 256], color='gray')
    axes[1].axvline(x=thresh, color='r', linestyle='--', linewidth=2)
    axes[1].set_title("Histograma de la Imagen en Grises")
    axes[1].set_xlabel("Intensidad de Píxel")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_xlim([0, 256])
    
    # --- Guardar y mostrar la figura ---
    fig.suptitle(f"Análisis de Binarización con Umbral = {thresh}", fontsize=16)
    save_path = f"plots/binarization_analysis_thresh_{thresh}.png"
    plt.savefig(save_path)
    print(f"Gráfico guardado en: {save_path}")
    plt.show()
    plt.close(fig)

print("\nProceso de binarización con múltiples umbrales completado.")

# 4. Combinar los 4 gráficos en una sola imagen
print("\nCombinando los 4 gráficos de análisis en una sola imagen...")
try:
    # Cargar las 4 imágenes generadas
    img1 = cv2.imread("plots/binarization_analysis_thresh_51.png")
    img2 = cv2.imread("plots/binarization_analysis_thresh_102.png")
    img3 = cv2.imread("plots/binarization_analysis_thresh_153.png")
    img4 = cv2.imread("plots/binarization_analysis_thresh_204.png")

    # Verificar si las imágenes se cargaron correctamente
    if any(img is None for img in [img1, img2, img3, img4]):
        raise FileNotFoundError("No se pudo cargar una o más de las imágenes de análisis. Asegúrate de que el script se ejecutó completamente antes.")

    # Crear la primera fila de la cuadrícula
    top_row = np.hstack((img1, img2))
    # Crear la segunda fila de la cuadrícula
    bottom_row = np.hstack((img3, img4))
    
    # Combinar las dos filas verticalmente
    combined_image = np.vstack((top_row, bottom_row))
    
    # Guardar la imagen combinada
    combined_save_path = "plots/combined_binarization_analysis.png"
    cv2.imwrite(combined_save_path, combined_image)
    
    print(f"Imagen combinada guardada en: {combined_save_path}")
    
    # Mostrar la imagen final (usando matplotlib para mejor visualización)
    plt.figure(figsize=(20, 10))
    # OpenCV carga en BGR, matplotlib muestra en RGB. Se necesita conversión.
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    # plt.title("Análisis Combinado de Binarización", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")


