# # Procesamiento de imágenes (pre TP2)
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import numpy as np
import urllib.request
from sklearn.cluster import KMeans
# ## Las  imágenes son básicamente matrices
# Mínimo elemento de la matriz se conoce como pixel.
# Cada pixel es un dígito está entre (0,255) *(8 bits)* 

# ## Las imágenes a color
# 1 dimensión más  ⟶ RGB
# (En el notebook original, hay una imagen que ilustra los canales RGB)
urllib.request.urlretrieve("https://i.ytimg.com/vi/M7o3-En283c/maxresdefault.jpg", "paisaje.jpg")
img = cv2.imread('paisaje.jpg')
print(img.shape)
plt.imshow(img)
plt.show()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

# ## Juntamos las 2 imagenes para que se aprecie la diferencia
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.set_title('BGR')
ax1.imshow(img)
ax2.set_title('RGB')
ax2.imshow(img_rgb)
plt.show()

# ## Re-escalar una imágen
# Reducir el tamaño de la imagen para que el algoritmo no demore tanto en ejecutarse.
img_reducida = cv2.resize(img_rgb, (300,200), interpolation=cv2.INTER_AREA)
plt.imshow(img_reducida)
plt.show()
print(img_reducida.shape)

# # Clustering
#
# El objetivo del clustering es agrupar data similar.
#
# En nuestro caso, queremos agrupar los pixeles de colores similares.
#
# Vamos a usar el algoritmo de [K-Means](https://es.wikipedia.org/wiki/K-medias) que es uno de los algoritmos más simples y populares.
#
# La idea es definir "K" centroides (pixeles de un color determinado) y agrupar el resto de los pixeles al centroide más cercano (con menor distancia).
dos_d = img_reducida.reshape(-1,3)
print(dos_d.shape)
# Ahora tenemos una matriz de 60000 pixeles x 3 (canales)

kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(dos_d)
print(kmeans.labels_)
print(np.unique(kmeans.labels_))
print(kmeans.cluster_centers_)

# Como podemos observar, los centroides son 5 pixeles con sus respectivos valores de R, G y B.
centroides = kmeans.cluster_centers_.astype('uint8')
print(centroides)

# ## Re-armamos la imágen en base a los centroides
etiquetas = kmeans.labels_
print(etiquetas.shape)
img_cluster = centroides[etiquetas]
print(img_cluster.shape)
img_cluster_2 = img_cluster.reshape(img_reducida.shape)
print(img_cluster_2.shape)
plt.imshow(img_cluster_2)
plt.show()

# ## Juntamos las 2 imagenes para ver el resultado final
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.set_title('Original')
ax1.imshow(img_reducida)
ax2.set_title('Clustering')
ax2.imshow(img_cluster_2)
plt.show()

# # Fin 
# 1 dimensión más  ⟶ RGB
# (En el notebook original, hay una imagen que ilustra los canales RGB)

## Jugando con las dimensiones
# Miremos las dimensiones del array que representa a la imagen
print("Dimensiones de la imagen a color:", img.shape)

# Tiene 3 dimensiones porque es una imagen a color (RGB)

# La podemos convertir a escala de grises para simplificarla
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Dimensiones de la imagen en escala de grises:", gray_img.shape)

# Ahora tiene 2 dimensiones

# Podemos ver la imagen en escala de grises
plt.imshow(gray_img, cmap='gray')
plt.title("Imagen en Escala de Grises")
plt.show()

## Clustering
# El algoritmo de clustering que vamos a usar es K-Means.

# K-means espera recibir como input un array de 2 dimensiones
# Dónde cada fila es un punto a agrupar
# Y cada columna es una dimensión del punto
print("Dimensiones de la imagen en escala de grises (pre-reshape):", gray_img.shape)

# Hacemos un reshape para que el array de la imagen se ajuste a lo que espera K-means
# Es decir, convertimos la matriz de 720x1280 en un array de 921600x1
# (720*1280 = 921600)
image_2d = gray_img.reshape(-1, 1)
print("Dimensiones de la imagen para K-means:", image_2d.shape)

from sklearn.cluster import KMeans

# Elegimos la cantidad de clusters (K)
# En este caso, al ser una imagen, K representa la cantidad de colores que tendrá la imagen segmentada
K = 8
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(image_2d)

### Segmentación de la imagen

# Obtenemos las etiquetas de cada pixel
labels = kmeans.labels_
print("Etiquetas de los pixeles (primeros 10):", labels[:10])

# Obtenemos los centroides de los clusters
# que en este caso representan los colores de la imagen segmentada
centroids = kmeans.cluster_centers_
print("Centroides (colores de la imagen segmentada):", centroids)

# Reemplazamos cada pixel por el centroide de su cluster
segmented_image_2d = centroids[labels]

# Hacemos un reshape para que vuelva a tener las dimensiones de la imagen original
segmented_image = segmented_image_2d.reshape(gray_img.shape)

# Mostramos la imagen original y la segmentada
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gray_img, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(segmented_image, cmap='gray')
axs[1].set_title('Segmentada')
plt.show()

## Elbow method
# Para elegir el K óptimo, podemos usar el método del codo (Elbow method), que consiste en graficar la inercia del modelo en función de K.
# La inercia es la suma de las distancias al cuadrado de cada punto a su centroide más cercano.
inertias = []
K_range = range(1, 10)
for K_val in K_range:
    kmeans = KMeans(n_clusters=K_val, random_state=42, n_init=10)
    kmeans.fit(image_2d)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias)
plt.xlabel('K')
plt.ylabel('Inercia')
plt.title('Elbow method')
plt.show()

# A partir de la imágen podemos ver que un buen K podría ser 3 ó 4, ya que a partir de ahí la curva se "aplana".

## Clustering sobre una imagen a color (3D)

# Hacemos un reshape para que el array de la imagen se ajuste a lo que espera K-means
# Es decir, convertimos la matriz de 720x1280x3 en un array de 921600x3
image_3d = img.reshape(-1, 3)
print("Dimensiones de la imagen a color para K-means:", image_3d.shape)

K = 8
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(image_3d)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
segmented_image_3d = centroids[labels].reshape(img.shape).astype(np.uint8)

# Mostramos la imagen original y la segmentada
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')
axs[1].imshow(cv2.cvtColor(segmented_image_3d, cv2.COLOR_BGR2RGB))
axs[1].set_title('Segmentada')
plt.show()

# # Actividad
# 1.  Correr el notebook con otra imágen (puede ser una local o una de internet)
# 2.  Correr el notebook con otros valores de K
# 3.  Correr el notebook con otro algoritmo de clustering ([ver documentación de scikit-learn](https://scikit-learn.org/stable/modules/clustering.html)) 