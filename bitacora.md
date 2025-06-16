# Bitácora del Proyecto de Clustering

## 2025-06-16 - Reemplazo de Filtro de Nitidez por Filtro Sobel

### Acciones Implementadas:
- Se ha modificado el script `5_3_filters.py` para reemplazar el filtro de realce de nitidez (sharpening) por el filtro de detección de bordes de Sobel.
- El script ahora demuestra cómo aplicar los filtros de Sobel en las direcciones horizontal (X) y vertical (Y).
- Antes de aplicar Sobel, la imagen se convierte a escala de grises, ya que el operador Sobel trabaja sobre imágenes de un solo canal.
- Se ha actualizado la salida del script para mostrar una cuadrícula de 2x2 con:
    1. La imagen original a color.
    2. La imagen en escala de grises.
    3. El resultado del filtro Sobel horizontal.
    4. El resultado del filtro Sobel vertical.
- Se ha actualizado la explicación en la salida de la consola para describir el funcionamiento y los casos de uso del filtro de Sobel.

### Contexto y Consideraciones:
- El filtro de Sobel es una técnica fundamental en el preprocesamiento de imágenes para la detección de características, específicamente para encontrar bordes.
- Mostrar los resultados de los ejes X e Y por separado permite entender cómo el filtro detecta gradientes de intensidad en diferentes orientaciones.
- Este cambio enriquece el conjunto de herramientas de análisis de imágenes del proyecto, pasando de un simple realce a una técnica de extracción de características más avanzada.

## 2025-06-15 - Inicialización y Análisis del Proyecto

### Acciones Implementadas:
- Se han estudiado los ficheros iniciales del proyecto: `DMCyT 2025 pre TP2.pdf` y `dmcyt_2025_tutorial_pretp2_clustering.py`.
- El script de Python (`dmcyt_2025_tutorial_pretp2_clustering.py`) es un tutorial introductorio al procesamiento de imágenes y clustering. Utiliza librerías como OpenCV, Matplotlib, Pandas y NumPy. El script parece ser una conversión de un notebook de Colab.
- Se ha creado este fichero `bitacora.md` para registrar los cambios relevantes y las decisiones tomadas a lo largo del proyecto, siguiendo las directrices establecidas.

### Contexto y Consideraciones:
- Este es el punto de partida para el trabajo práctico de la asignatura "DMCyT 2025".
- La bitácora servirá como un diario de proyecto para documentar el progreso, los experimentos y las conclusiones.

## 2025-06-15 - Conversión de Notebook a Script de Python

### Acciones Implementadas:
- Se ha convertido el notebook `Copia_de_DMCyT_2025_Tutorial_preTP2_clustering.ipynb` a un script de Python puro llamado `dmcyt_2025_tutorial_pretp2_clustering_from_notebook.py`.
- Se han añadido comentarios al script para explicar cada paso del proceso de clustering de imágenes.
- Los comandos de shell del notebook (como `wget`) han sido reemplazados por código Python equivalente (`urllib.request`).
- Se han añadido llamadas a `print()` y `plt.show()` para asegurar que el script produce resultados visibles al ser ejecutado.

### Contexto y Consideraciones:
- La conversión a un script de Python puro facilita la ejecución del código en entornos que no soportan notebooks y permite una mejor integración en flujos de trabajo de desarrollo de software.
- El script resultante es más legible y mantenible gracias a la adición de comentarios y la reestructuración del código.

## 2025-06-15 - Creación del fichero `requirements.txt`

### Acciones Implementadas:
- Se han analizado los scripts de Python del proyecto (`dmcyt_2025_tutorial_pretp2_clustering.py` y `dmcyt_2025_tutorial_pretp2_clustering_from_notebook.py`) para identificar las dependencias.
- Se ha creado el fichero `requirements.txt` con las siguientes librerías: `opencv-python`, `matplotlib`, `pandas`, `numpy` y `scikit-learn`.

### Contexto y Consideraciones:
- La creación de `requirements.txt` es un paso fundamental para asegurar la reproducibilidad del entorno de desarrollo.
- Este fichero permitirá a cualquier desarrollador instalar fácilmente todas las dependencias necesarias para ejecutar los scripts del proyecto con el comando `pip install -r requirements.txt`.

## 2024-05-18
- Se analiza el proyecto y se crea el archivo `bitacora.md`.
- Se identifica que el script `dmcyt_2025_tutorial_pretp2_clustering.py` es un tutorial introductorio a procesamiento de imágenes y clustering, probablemente convertido desde un notebook de Colab.

## 2024-05-19
- Se convierte el Jupyter Notebook `Copia_de_DMCyT_2025_Tutorial_preTP2_clustering.ipynb` a un script de Python puro llamado `dmcyt_2025_tutorial_pretp2_clustering_from_notebook.py`.
- Se añaden comentarios al script para explicar cada paso del proceso.
- La conversión se realiza de manera incremental, asegurando que el script sea funcional y fácil de entender.
- Se reemplazan las salidas de celda del notebook (como mostrar `shape` o gráficos) con sentencias `print()` y `plt.show()` para que el script sea ejecutable y muestre resultados en la consola y en ventanas de gráficos. 

## 2025-06-15 - Creación de Scripts para el pre-TP2

### Acciones Implementadas:
- Se han creado scripts de Python individuales para resolver cada uno de los puntos del enunciado del pre-TP2 (`DMCyT 2025 pre TP2.html`).
- Los scripts han sido nombrados siguiendo una convención numérica basada en los puntos del enunciado (e.g., `4_1_load_data.py`, `5_2_random_images.py`, etc.).
- Cada script es autocontenido y realiza una tarea específica, desde la carga de datos hasta el análisis con PCA.
- Los scripts incluyen:
    - **4_1_load_data.py**: Carga de datos y verificación inicial.
    - **4_2_explore_data.py**: Visualización de imágenes por especie.
    - **5_1_grayscale_binary.py**: Conversión a escala de grises y binarización, con explicación.
    - **5_2_random_images.py**: Generación de imágenes aleatorias (píxeles mezclados y composición).
    - **5_3_filters.py**: Aplicación y explicación de filtros Gaussiano y de nitidez.
    - **5_4_average_images.py**: Cálculo de imágenes promedio (global y por especie).
    - **6_1_pixel_distribution.py**: Análisis de distribución de color por especie.
    - **6_2_pca.py**: Análisis de Componentes Principales para visualización de la separabilidad de las especies.

### Contexto y Consideraciones:
- La creación de scripts modulares y específicos para cada tarea facilita la ejecución, prueba y depuración de cada parte del trabajo práctico.
- Este enfoque permite centrarse en un problema a la vez y hace que el código sea más legible y mantenible.
- Cada script incluye comentarios y salidas visuales (gráficos) para ilustrar los resultados, siguiendo las mejores prácticas de análisis exploratorio de datos.

## 2025-06-16 - Ajuste de Imágenes y Creación de Grilla de Especies

### Acciones Implementadas:
- Se ha modificado el script `4_1_find_mismatched_size_images.py`. Ahora, en lugar de solo reportar las imágenes con dimensiones incorrectas, las redimensiona automáticamente a 128x128 píxeles. La interpolación `cv2.INTER_AREA` fue seleccionada por ser la más adecuada para reducir el tamaño de las imágenes, ya que promedia los píxeles del área original, conservando mejor la calidad. El script sobrescribe las imágenes originales.
- Se ha actualizado el script `4_2_explore_data.py` para generar una visualización consolidada. Ahora, guarda una imagen de muestra para cada una de las 10 especies en el directorio `plots/`. Adicionalmente, crea una única figura con una cuadrícula de 3x4 que muestra estas 10 imágenes y la guarda como `flower_species_grid.png` en `plots/`.

### Contexto y Consideraciones:
- La estandarización del tamaño de las imágenes es un paso de preprocesamiento crucial para muchos algoritmos de machine learning, que requieren que los datos de entrada tengan dimensiones consistentes. La elección de `cv2.INTER_AREA` busca minimizar la pérdida de información durante el reescalado.
- La creación de una grilla de imágenes de especies permite una rápida verificación visual de la diversidad del dataset y facilita la comunicación de los resultados en informes o presentaciones. 