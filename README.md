# Vision-ST
Laboratorio de vision para la semana Tec: Herramientas computacionales: el arte de la programación

## Creadores:
- Clarissa Gardea Coronado
- Antonino Sandoval

Este repositorio contiene un proyecto de laboratorio cuyo objetivo es detectar números de placas de automóviles mediante procesamiento de imágenes y reconocimiento óptico de caracteres (OCR). Utilizando Python, OpenCV, NumPy y Tesseract OCR, el script aplica técnicas de convolución, padding y saturación de imágenes para mejorar la precisión en la detección de texto.

## Librerias utilizadas: 
- [NumPy](https://numpy.org/) para manejo de arreglos y cálculos numéricos.
- [OpenCV](https://opencv.org/) para operaciones de procesamiento de imágenes.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) para reconocimiento de texto.
- [Matplotlib](https://matplotlib.org/) para visualización de imágenes.

##  Descripción de los Métodos
### 1. Convolución
La convolución se utiliza para aplicar un filtro específico a la imagen con el objetivo de resaltar los caracteres de las placas. Este método consiste en multiplicar un área de la imagen por un kernel, lo cual permite mejorar la nitidez y el contraste en regiones específicas, facilitando la identificación de los números.

### 2. Padding
El padding añade un borde de píxeles a la imagen, permitiendo aplicar la convolución sin perder detalles en los bordes. De esta forma, las características que se encuentran cerca de los márgenes de la imagen se procesan adecuadamente, lo cual es especialmente útil para detección en regiones periféricas de las placas.

### 3. Saturación
La saturación ajusta la intensidad de los píxeles para mejorar el contraste general de la imagen. En este caso, se aplica un factor de contraste que realza los números en la placa y reduce la interferencia del fondo, aumentando la precisión del OCR.

## Codigos en el repositorio:
### sobel.py
Implementa un flitro sobel como metodo de convolucion para una mejor deteccion de bordes

### gaussian_blurr.py
Implememnta un filtro gaussiano como metodo de convolucion para un degradado en la imagen

### simple_conv.py
Implementacion de una convolucion simple con dos filtros sobel, uno para bordes horizontales y otro para bordes verticales.

### convolution.py
Implementacion de una convolucion con padding para una mejor aplicacion del filtrado, este codigo tambien trabaja con dos filtros sobel, para cada uno para bordes verticales o horizontales.

### convolucionFinal.py
Codigo final con la implementacion del OCR con la libreria Tesseract para no solo aplicar filtrado si no tambien tener la capacidad de convertir los caracteres de las letras filtradas en la placa a texto en consola.


## Uso
Carga de imagen: El script carga una imagen en escala de grises desde una ruta especificada. Actualiza la ruta en el código (image = cv2.imread(r'D:/path_to_your_image/image.jpg')) con la ubicación de tu imagen.

### Ejecución del procesamiento de imagen:

La función convolution() aplica el kernel de convolución, padding y saturación para realzar los caracteres.
La imagen resultante es mostrada y, finalmente, analizada mediante OCR para detectar texto.
OCR para detección de números de placas:

Tesseract OCR realiza un reconocimiento de caracteres en la imagen procesada.
Los caracteres detectados se imprimen en la consola.



Este README proporciona una descripción general del proyecto, instrucciones de instalación, uso y detalles sobre el proceso de convolución y OCR.
