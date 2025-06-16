import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- 5.3 Aplicar diferentes tipos de filtros ---

# Directorio de imágenes
image_dir = "kaggle_flower_images"
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])

# Seleccionar una imagen de ejemplo
sample_image_path = image_files[20] # Usar otra imagen
original_image = cv2.imread(sample_image_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)


# --- Filtro 1: Desenfoque Gaussiano (Gaussian Blur) ---
# Kernel size (ksize) debe ser una tupla de números impares positivos.
# A mayor ksize, mayor es el desenfoque.
gaussian_blurred = cv2.GaussianBlur(original_image, (15, 15), 0)
gaussian_blurred_rgb = cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2RGB)

print("""
--- Filtro 1: Desenfoque Gaussiano (Gaussian Blur) ---
Este filtro suaviza la imagen promediando los valores de los píxeles vecinos con una 
ponderación que sigue una distribución gaussiana. Los píxeles más cercanos al centro 
del "kernel" tienen más influencia.

¿Cuándo conviene usarlo?
- Reducción de ruido: Es muy efectivo para eliminar ruido de alta frecuencia (como el ruido "sal y pimienta" o el ruido gaussiano) sin afectar demasiado los bordes.
- Pre-procesamiento: Se usa a menudo antes de otras operaciones como la detección de bordes (ej. Canny), ya que al suavizar la imagen se evitan falsos bordes causados por el ruido.
- Efectos artísticos: Para crear un efecto de desenfoque o "bokeh".
----------------------------------------------------
""")

# Graficar el efecto del filtro Gaussiano
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image_rgb)
plt.title("Imagen Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gaussian_blurred_rgb)
plt.title("Filtro Gaussiano Aplicado")
plt.axis('off')

plt.suptitle("Desenfoque Gaussiano", fontsize=16)
plt.show()


# --- Filtro 2: Aumento de Nitidez (Sharpening) ---
# Se puede lograr creando un "kernel" de realce de bordes.
# Un kernel común es el que resta los vecinos del píxel central, amplificando las diferencias.
sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])

sharpened_image = cv2.filter2D(original_image, -1, sharpening_kernel)
sharpened_image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

print("""
--- Filtro 2: Aumento de Nitidez (Sharpening) ---
Este filtro realza los bordes y detalles finos en una imagen. Funciona acentuando 
la diferencia de intensidad entre un píxel y sus vecinos. El "kernel" utilizado
generalmente tiene un valor central positivo alto y valores negativos a su alrededor.

¿Cuándo conviene usarlo?
- Mejorar detalles: Cuando una imagen parece ligeramente borrosa o desenfocada, este filtro puede hacer que los detalles sean más nítidos y claros.
- Análisis de texturas: Ayuda a resaltar las texturas y patrones finos en una imagen.
- Post-procesamiento: Comúnmente usado en fotografía para dar un toque final de nitidez a las imágenes.
- Advertencia: Puede amplificar el ruido existente en la imagen, por lo que a veces se aplica después de un filtro de reducción de ruido suave.
----------------------------------------------------
""")

# Graficar el efecto del filtro de nitidez
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image_rgb)
plt.title("Imagen Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sharpened_image_rgb)
plt.title("Filtro de Nitidez Aplicado")
plt.axis('off')

plt.suptitle("Aumento de Nitidez", fontsize=16)
plt.show() 