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


# --- Filtro 2: Detección de Bordes con Sobel ---
# Convertir a escala de grises para la detección de bordes
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Aplicar filtro de Sobel en la dirección X (bordes verticales)
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_x_abs = np.absolute(sobel_x)
sobel_x_uint8 = np.uint8(sobel_x_abs)

# Aplicar filtro de Sobel en la dirección Y (bordes horizontales)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_y_abs = np.absolute(sobel_y)
sobel_y_uint8 = np.uint8(sobel_y_abs)

print("""
--- Filtro 2: Detección de Bordes con Sobel ---
El filtro de Sobel es un operador diferencial que se utiliza para la detección de
bordes en una imagen. Calcula el gradiente de la intensidad de la imagen en cada
píxel, lo que permite identificar áreas donde hay un cambio brusco en la intensidad,
característico de un borde.

Funciona con dos kernels (máscaras), uno para detectar bordes horizontales y otro
para bordes verticales.

- Sobel Horizontal (Sobel X): Detecta cambios de intensidad en la dirección horizontal, lo que resalta los bordes verticales.
- Sobel Vertical (Sobel Y): Detecta cambios de intensidad en la dirección vertical, lo que resalta los bordes horizontales.

¿Cuándo conviene usarlo?
- Detección de características: Es fundamental en la visión por computadora para extraer características como bordes y contornos.
- Segmentación de imágenes: Puede ser un paso previo para segmentar objetos en una imagen.
- Análisis de formas: Al resaltar los contornos, facilita el análisis de la forma de los objetos.
----------------------------------------------------
""")

# Graficar el efecto del filtro de Sobel
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.imshow(original_image_rgb)
plt.title("Imagen Original")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title("Imagen en Escala de Grises")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(sobel_x_uint8, cmap='gray')
plt.title("Filtro Sobel Horizontal (X)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(sobel_y_uint8, cmap='gray')
plt.title("Filtro Sobel Vertical (Y)")
plt.axis('off')

plt.suptitle("Detección de Bordes con Sobel", fontsize=16)
plt.show() 