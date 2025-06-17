import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters

# --- 5.3 Aplicar diferentes tipos de filtros ---

# Directorio de imágenes
image_dir = "kaggle_flower_images"
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])

# Seleccionar una imagen de ejemplo
sample_image_path = os.path.join(image_dir, "0007.png") # Usar imagen 0007.png
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


# --- Filtro 2: Detección de Bordes con Sobel (usando scikit-image) ---
# Convertir a escala de grises para la detección de bordes
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Aplicar los filtros de Sobel usando las funciones de alto nivel de scikit-image
# sobel_v -> Detecta bordes verticales (derivada en X)
# sobel_h -> Detecta bordes horizontales (derivada en Y)
sobel_v = filters.sobel_v(gray_image)
sobel_h = filters.sobel_h(gray_image)


print("""
--- Filtro 2: Detección de Bordes con Sobel (usando scikit-image) ---
El filtro de Sobel es un operador diferencial que se utiliza para la detección de
bordes en una imagen. Calcula el gradiente de la intensidad de la imagen en cada
píxel, lo que permite identificar áreas donde hay un cambio brusco en la intensidad,
característico de un borde.

En este caso, se utilizan las funciones `sobel_h` y `sobel_v` de la librería
scikit-image, que son implementaciones directas y optimizadas de este filtro.

- Sobel Vertical (sobel_v): Detecta cambios de intensidad en la dirección horizontal, resaltando los bordes verticales.
- Sobel Horizontal (sobel_h): Detecta cambios de intensidad en la dirección vertical, resaltando los bordes horizontales.

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
plt.imshow(sobel_v, cmap='gray')
plt.title("Filtro Sobel Vertical (scikit-image)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(sobel_h, cmap='gray')
plt.title("Filtro Sobel Horizontal (scikit-image)")
plt.axis('off')

plt.suptitle("Detección de Bordes con Sobel (usando scikit-image)", fontsize=16)
plt.show()


# --- Filtro 3: Realce de Todos los Bordes (con valores bajos) ---

# 1. Calcular la magnitud del gradiente combinando los ejes X e Y.
#    La función sobel() sin sufijo hace esto automáticamente.
edge_magnitude = filters.sobel(gray_image)

# 2. Normalizar la magnitud al rango [0, 1] para una inversión predecible.
edge_magnitude_normalized = edge_magnitude / np.max(edge_magnitude)

# 3. Invertir la imagen. Los bordes fuertes (cercanos a 1) se volverán oscuros (cercanos a 0).
inverted_edges = 1 - edge_magnitude_normalized

print("""
--- Filtro 3: Realce de Todos los Bordes (con valores bajos) ---
Para realzar todos los bordes con un valor de píxel bajo (hacerlos oscuros),
se siguen los siguientes pasos:
1.  Se calcula la magnitud total del gradiente usando `skimage.filters.sobel()`.
    Esto combina las componentes horizontal y vertical en una sola imagen que
    representa la "fuerza" de cada borde.
2.  La imagen de magnitud se normaliza al rango [0, 1], donde 1 es el borde
    más fuerte.
3.  Finalmente, se invierten los valores restando la imagen normalizada de 1.
    Esto hace que los bordes fuertes (antes 1) se conviertan en 0 (negro),
    y las áreas sin bordes (antes 0) se conviertan en 1 (blanco).
----------------------------------------------------
""")

# Graficar el resultado del realce de bordes
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image_rgb)
plt.title("Imagen Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(inverted_edges, cmap='gray')
plt.title("Bordes Realzados (Oscuros)")
plt.axis('off')

plt.suptitle("Realce Invertido de Bordes", fontsize=16)
plt.show() 