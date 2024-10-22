"""
By Abhisek Jana
code taken from https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
blog http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
"""
#gaussian_blur.py

import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import math
from convolution import convolution

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    
    kernel_2D *= 1.0 / kernel_2D.max()

    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()

    return kernel_2D

def gaussian_blur(image, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    return convolution(image, kernel, average=True, verbose=verbose)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="Path to the image")
    args = vars(ap.parse_args())

    # Check if the image path is provided, if not use a default
    if args["image"] is None:
        image_path = r"D:\ESCUELA\Laboratorio\Vision-ST\image.jpg"  # Default image path
    else:
        image_path = args["image"]

    print(f"Attempting to load image from: {image_path}")  # Debugging step

    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Apply Gaussian Blur
    gaussian_blur(image, 5, verbose=True)

