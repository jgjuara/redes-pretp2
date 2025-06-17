# Bitácora del Proyecto de Clustering

## 2025-06-17 - Configuración del Reporte para el Estándar NeurIPS

### Acciones Implementadas:
- Se ha modificado el fichero `report.qmd` para que el formato de salida PDF se ajuste al estándar de la conferencia NeurIPS.
- Se ha eliminado la configuración manual de formato en el YAML (geometría, tipo y tamaño de fuente, interlineado).
- En su lugar, se ha añadido una directiva para incluir el paquete de LaTeX `neurips_2022.sty` que se encuentra en el proyecto.

### Contexto y Consideraciones:
- El objetivo es que el reporte final tenga un formato profesional y consistente con los estándares de publicaciones científicas en el área.
- Utilizar el paquete `neurips_2022.sty` asegura que todos los elementos del documento (títulos, secciones, texto, etc.) sigan las guías de estilo de NeurIPS.
- Este cambio centraliza la gestión del formato en el fichero `.sty`, haciendo el `report.qmd` más limpio y enfocado en el contenido.

## 2025-06-17 - Corrección del Tamaño de Figura en Gráfico de Distribución

### Acciones Implementadas:
- Se ha modificado el script `6_3_pixel_distribution_grays.py` para corregir la visualización de los títulos de los subgráficos.
- Se redujo la altura total de la figura, cambiando el parámetro `figsize` en `plt.subplots`. La altura ahora se calcula como `2 * len(species)` en lugar de `6 * len(species)`.
- Se eliminó el ajuste manual de la posición `y` del título, ya que el problema de fondo era el tamaño excesivo del subgráfico y no la posición por defecto del título.

### Contexto y Consideraciones:
- Los títulos de los subgráficos aparecían "flotando" muy por encima de las curvas de densidad. La causa raíz era una altura de figura desproporcionadamente grande (`12x60` pulgadas), lo que hacía que cada subgráfico fuera muy alto.
- Al reducir la altura de cada subgráfico, la posición por defecto del título ahora es visualmente correcta y está apropiadamente cerca de la gráfica que describe. Esto mejora significativamente la legibilidad y la estética del gráfico.

## 2025-06-17 - Optimización de Rendimiento del Script de Distribución

### Acciones Implementadas:
- Se ha modificado el script `6_3_pixel_distribution_grays.py` para solucionar un problema de rendimiento que lo hacía parecer "colgado".
- Se reemplazó la función `sns.kdeplot` por `sns.histplot` con 256 `bins`. El cálculo de histogramas es computacionalmente mucho más eficiente que la estimación de densidad por kernels (KDE) para conjuntos de datos grandes.
- Se añadió un indicador de progreso en la consola para informar al usuario sobre qué especie se está procesando.
- Se optimizó la carga de imágenes leyéndolas directamente en escala de grises (`cv2.IMREAD_GRAYSCALE`) en lugar de convertirlas después de la carga.

### Contexto y Consideraciones:
- El script procesa todos los píxeles de todas las imágenes, lo que resulta en un gran volumen de datos. La función `kdeplot` era demasiado lenta para esta tarea.
- La nueva implementación con `histplot` es significativamente más rápida y aún permite una visualización efectiva de la distribución de la intensidad de los píxeles.
- Las optimizaciones adicionales (indicador de progreso y carga directa en gris) mejoran la experiencia del usuario y la eficiencia del script.

## 2025-06-17 - Corrección de Etiquetas en Gráfico de Distribución

### Acciones Implementadas:
- Se ha modificado el script `6_3_pixel_distribution_grays.py` para mejorar la legibilidad del gráfico de distribución de píxeles en escala de grises.
- Se eliminaron las etiquetas de los ejes X e Y de cada subgráfico individual.
- Se añadieron etiquetas centrales (`supxlabel` y `supylabel`) para toda la figura, proporcionando una descripción clara y única para los ejes compartidos.

### Contexto y Consideraciones:
- El comportamiento por defecto de Matplotlib con ejes compartidos (`sharex`, `sharey`) es ocultar las etiquetas de los ejes internos para evitar redundancia, lo que podía hacer que el gráfico pareciera incompleto.
- La centralización de las etiquetas mejora la estética y la claridad del gráfico, haciendo más fácil su interpretación.

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

## 2025-06-17 - Análisis de Distribución de Píxeles en Imágenes en Escala de Grises

### Acciones Implementadas:
- Se ha modificado el script `6_3_pixel_distribution_binary.py` para cambiar el análisis. En lugar de analizar imágenes binarizadas, ahora analiza la distribución de intensidad de píxeles en escala de grises.
- Para cada especie, el script:
    1. Carga todas las imágenes correspondientes.
    2. Convierte cada imagen a escala de grises.
    3. Acumula todos los valores de los píxeles (0-255) de las imágenes en escala de grises.
- Se genera un gráfico de densidad (KDE), con un subgráfico por especie, mostrando la distribución de luminosidad.
- El gráfico resultante se guarda en `plots/6_3_pixel_distribution_grayscale.png`.

### Contexto y Consideraciones:
- Se determinó que analizar la distribución de una imagen binarizada (con solo dos valores posibles) era poco informativo, ya que solo muestra la proporción entre píxeles de primer y segundo plano.
- El nuevo análisis sobre la escala de grises es más detallado. Permite observar la distribución completa de la luminosidad de las flores de cada especie, independientemente de su color.
- Este enfoque puede revelar patrones en el brillo y contraste de las flores que podrían ser útiles para la clasificación, complementando el análisis de los canales de color individuales.

## 2025-06-17 - Análisis de Varianza Explicada en PCA

### Acciones Implementadas:
- Se ha modificado el script `6_2_pca.py` para añadir un análisis de la varianza explicada por los componentes principales.
- Antes de la visualización de 2D, el script ahora realiza un PCA con todos los componentes posibles.
- Se ha añadido un nuevo gráfico que muestra la curva de varianza explicada acumulada en función del número de componentes.
- El gráfico incluye una línea de referencia horizontal para el 95% de la varianza, y el script imprime en consola el número exacto de componentes necesarios para alcanzar este umbral.
- Se ha añadido una línea de referencia vertical en `n=2` componentes para visualizar directamente cuánta varianza es capturada por la representación 2D, junto con una etiqueta que muestra el valor exacto.

### Contexto y Consideraciones:
- La visualización de PCA en solo dos dimensiones es útil para una inspección rápida, pero a menudo captura una porción muy pequeña de la varianza total de los datos, lo que puede ser engañoso.
- El gráfico de varianza acumulada, ahora enriquecido con las líneas de referencia, permite cuantificar de forma precisa cuánta información se "pierde" al reducir la dimensionalidad a dos componentes.
- Ayuda a determinar el número óptimo de componentes a retener para futuros modelos de machine learning, buscando un equilibrio entre la reducción de la complejidad y la conservación de la información de los datos.
- Este análisis proporciona una comprensión más profunda de la estructura intrínseca del conjunto de datos de imágenes.

## 2025-06-17 - Generación de Gráfico PCA Interactivo

### Acciones Implementadas:
- Se ha modificado el script `6_2_pca.py` para generar una versión interactiva del gráfico de dispersión de PCA.
- Se añadió la librería `plotly` al fichero `requirements.txt` como una nueva dependencia del proyecto.
- El script ahora produce dos gráficos de PCA:
    1. La versión estática original, guardada como imagen (`pca_plot.png`).
    2. Una nueva versión interactiva creada con `plotly.express`, que se guarda como un fichero HTML (`interactive_pca_plot.html`) y se abre automáticamente en un navegador.
- En el gráfico interactivo, al pasar el cursor sobre un punto de datos, ahora se muestra la ruta del fichero de la imagen correspondiente.

### Contexto y Consideraciones:
- Un gráfico interactivo permite una exploración de datos mucho más rica que uno estático. Es posible hacer zoom en clusters de interés, filtrar datos por categoría haciendo clic en la leyenda y obtener información específica de puntos de datos individuales.
- La capacidad de ver el nombre del fichero de la imagen al pasar el cursor es especialmente útil para identificar rápidamente ejemplos específicos, como outliers o puntos en la frontera entre dos clusters, facilitando un análisis más profundo. 