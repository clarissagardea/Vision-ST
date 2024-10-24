"""
By Abhisek Jana
code taken from https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
blog http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
"""
import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt

# Ruta de instalación de Tesseract (ajusta esto según tu instalación)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    return output

# Cargar la imagen usando OpenCV
image = cv2.imread(r'D:/ESCUELA/Laboratorio/Vision-ST/images/image.jpg', cv2.IMREAD_GRAYSCALE)

# Aumentar el contraste multiplicando por un factor (ejemplo: 1.5)
image = np.clip(image * 1.5, 0, 255).astype(np.uint8)

# Mostrar la imagen original
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.show()

# Definir un kernel (Blur más potente)
kernel4 = np.array([[0, 0, 0, 0, 10, 0, 0, 0, 0],
                           [0, 0, 10, 25, 50, 25, 10, 0, 0],
                           [0, 10, 50, 100, 150, 100, 50, 10, 0],
                           [0, 25, 100, 150, 200, 150, 100, 25, 0],
                           [10, 50, 150, 200, 250, 200, 150, 50, 10],
                           [0, 25, 100, 150, 200, 150, 100, 25, 0],
                           [0, 10, 50, 100, 150, 100, 50, 10, 0],
                           [0, 0, 10, 25, 50, 25, 10, 0, 0],
                           [0, 0, 0, 0, 10, 0, 0, 0, 0]])

# Aplicar la convolución con el kernel definido
output_image = convolution(image, kernel4, average=True)

# Aumentar el contraste multiplicando por un factor (ejemplo: 0.1)
output_image = np.clip(output_image * 0.1, 0, 255).astype(np.uint8)

# Mostrar la imagen con los números grandes detectados
plt.imshow(output_image, cmap='gray')
plt.title("Números grandes detectados")
plt.show()

# Aplicar OCR para detectar letras y números
custom_config = r'--oem 3 --psm 6'
detected_text = pytesseract.image_to_string(output_image, config=custom_config)

# Imprimir las letras y números detectados en la consola
print("Letras detectadas:")
print(detected_text)

