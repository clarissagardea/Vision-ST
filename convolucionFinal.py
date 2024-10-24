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
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))
 
    print("Kernel Shape : {}".format(kernel.shape))
 
    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()
 
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
 
    output = np.zeros(image.shape)
 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
 
    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()
 
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
 
    print("Output Image size : {}".format(output.shape))
 
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()
 
    return output

# Cargar la imagen usando OpenCV
image = cv2.imread(r'c:\Users\anton\Downloads\Prueba5.jpg', cv2.IMREAD_GRAYSCALE)
# Aumentar el contraste multiplicando por un factor (ejemplo: 1.5)
image = np.clip(image * 1.5, 0, 255).astype(np.uint8)


# Mostrar la imagen original
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.show()

# Definir un kernel (Sobel X en este caso)
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

kernel2 = np.array([[0,0,0,5,0,0,0],
                    [0,5,18,32,18,5,0],
                    [0,18,64,100,64,18,0],
                    [5,32,100,100,100,32,5],
                    [0,18,64,100,64,18,0],
                    [0,5,18,32,18,5,0],
                    [0,0,0,5,0,0,0]])
#Blur mas potente
kernel3 = np.array([[0,0,0,0,5,0,0,0,0],
                    [0,0,5,18,32,5,0,0,0],
                    [0,0,18,64,100,64,18,0,0],
                    [0,5,64,100,100,100,64,5,0],
                    [5,18,100,100,100,100,100,18,5],
                    [0,5,64,100,100,100,64,5,0],
                    [0,0,18,64,100,64,18,0,0],
                    [0,0,5,18,32,5,0,0,0],
                    [0,0,0,0,5,0,0,0,0]])
                    # Blur más potente
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
output_image = convolution(image, kernel4,1,1)
output_image = convolution(image, kernel3,0,1)
#output_image = convolution(image, kernel3,0,1)

# Aumentar el contraste multiplicando por un factor (ejemplo: 1.5)
output_image = np.clip(output_image * .1, 0, 255).astype(np.uint8)
#output_image = convolution(output_image, kernel2,1,1)

#output_image2 = convolution(output_image, kernel,0,1)


# Mostrar la imagen original y la imagen con los números grandes detectados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Imagen original")

plt.subplot(1, 2, 2)
plt.imshow(output_image, cmap='gray')
plt.title("Números grandes detectados")
plt.show()
# Aplicar OCR para detectar texto en la imagen
custom_config = r'--oem 3 --psm 6 outputbase digits'  # Solo detectar dígitos
detected_text = pytesseract.image_to_string(output_image)

# Imprimir los números detectados en la consola
print("Números detectados:")
print(detected_text)