import cv2
import matplotlib.pyplot as plt
import os

# --- Comparación de Métodos de Umbralización (Thresholding) ---

# Cargar una imagen de ejemplo
image_dir = "kaggle_flower_images"
sample_image_path = os.path.join(image_dir, "0007.png")
image = cv2.imread(sample_image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- Método 1: Umbralización Global (cv2.threshold) ---
# Se usa un único valor de umbral (127) para toda la imagen.
_, global_thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# --- Método 2: Umbralización Adaptativa (cv2.adaptiveThreshold) ---
# El umbral se calcula para cada píxel basándose en un área local.
# blockSize = 11: El área de vecindad es de 11x11 píxeles.
# C = 2: Una constante que se resta de la media para afinar el umbral.
adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

print("""
--- Comparación de Métodos de Umbralización ---
1. Umbralización Global (cv2.threshold):
   - Usa un solo valor de umbral para toda la imagen.
   - Rápido y simple, pero ineficaz si la iluminación varía.

2. Umbralización Adaptativa (cv2.adaptiveThreshold):
   - Calcula un umbral diferente para cada píxel basado en su vecindad.
   - Mucho más robusto frente a cambios de iluminación.
   - Ideal para separar objetos del fondo en condiciones de luz no uniformes.
----------------------------------------------------
""")

# Mostrar los resultados
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(global_thresh, cmap='gray')
plt.title("Umbral Global (Threshold)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title("Umbral Adaptativo (Adaptive Threshold)")
plt.axis('off')

plt.suptitle("Comparación de Umbralización Global vs. Adaptativa", fontsize=16)
plt.show() 