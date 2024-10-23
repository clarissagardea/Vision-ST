"""
By Abhisek Jana
code taken from https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
blog http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/

Modified by Benjamin Valdes
"""
#Importar librerias
import numpy as np
import cv2 #Libreria de OpenCV para convoluciones
import matplotlib.pyplot as plt
 
def conv_helper(fragment, kernel):
    """ multiplica 2 matices y devuelve su suma"""
    
    f_row, f_col = fragment.shape
    k_row, k_col = kernel.shape 
    result = 0.0
    for row in range(f_row):
        for col in range(f_col):
            result += fragment[row,col] *  kernel[row,col]
    return result

<<<<<<< HEAD

=======
>>>>>>> 3d3bfcf08b4144a7f4b34273c978284f5d82c46b
def convolution(image, kernel):
    """Aplica una convolucion sin padding (valida) de una dimesion 
    y devuelve la matriz resultante de la operación
    """

    image_row, image_col = image.shape #asigna alto y ancho de la imagen 
    kernel_row, kernel_col = kernel.shape #asigna alto y ancho del filtro
   
    output = np.zeros(image.shape) #matriz donde guardo el resultado
   
    for row in range(image_row):
        for col in range(image_col):
                output[row, col] = conv_helper(
                                    image[row:row + kernel_row, 
                                    col:col + kernel_col],kernel)
             
    plt.imshow(output, cmap='gray')
    plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
    plt.show()
 
    return output

# Cargar la imagen usando OpenCV
image = cv2.imread(r'c:\Users\anton\Downloads\Prueba1.jpg', cv2.IMREAD_GRAYSCALE)

# Mostrar la imagen original
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.show()

# Definir un kernel (Sobel X en este caso)
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

kernel2 = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

# Aplicar la convolución con el kernel definido
output_image = convolution(image, kernel)
output_image2 = convolution(output_image, kernel2)