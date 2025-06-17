import cv2
import matplotlib.pyplot as plt
import os

# --- Efecto del parámetro C en la Umbralización Adaptativa ---

# Cargar una imagen de ejemplo
image_dir = "kaggle_flower_images"
sample_image_path = os.path.join(image_dir, "0007.png")
image = cv2.imread(sample_image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- Probar diferentes valores de C ---
# BlockSize se mantiene constante en 11 para aislar el efecto de C.

# C = 10: Un valor positivo grande. Reduce el umbral, resultando en más blanco.
adaptive_c10 = cv2.adaptiveThreshold(gray_image, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 10)

# C = 2: Un valor positivo pequeño (el que usamos antes).
adaptive_c2 = cv2.adaptiveThreshold(gray_image, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)

# C = -5: Un valor negativo. Aumenta el umbral, resultando en más negro.
adaptive_c_neg5 = cv2.adaptiveThreshold(gray_image, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, -5)

print("""
--- Efecto del Parámetro C ---
C es una constante que se resta de la media calculada en la vecindad.
- C > 0: El umbral baja, más píxeles se vuelven blancos. Útil para capturar detalles finos.
- C < 0: El umbral sube, más píxeles se vuelven negros. Útil para eliminar ruido.
----------------------------------------------------
""")

# Mostrar los resultados
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(adaptive_c10, cmap='gray')
plt.title("Umbral Adaptativo (C = 10)")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(adaptive_c2, cmap='gray')
plt.title("Umbral Adaptativo (C = 2)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(adaptive_c_neg5, cmap='gray')
plt.title("Umbral Adaptativo (C = -5)")
plt.axis('off')

plt.suptitle("Comparación de Diferentes Valores de C", fontsize=16)
plt.show() 