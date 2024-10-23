import numpy as np
import cv2
import matplotlib.pyplot as plt
from gaussian_blur import gaussian_blur
from sobel import sobel_edge_detection
import os

# Load predefined templates (0-9 digits and possibly letters if needed)
def load_templates(template_dir):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".png"):  # Assuming the templates are stored as PNG images
            template_char = filename.split('.')[0]  # Extract the character from the filename
            img = cv2.imread(os.path.join(template_dir, filename), cv2.IMREAD_GRAYSCALE)
            templates[template_char] = img
    return templates

# Match each segment with templates to identify characters
def match_segment_with_template(segment, templates):
    best_match = None
    best_score = float('inf')
    for char, template in templates.items():
        resized_template = cv2.resize(template, (segment.shape[1], segment.shape[0]))  # Resize template to match segment size
        result = cv2.matchTemplate(segment, resized_template, cv2.TM_SQDIFF)
        _, score, _, _ = cv2.minMaxLoc(result)  # Find the best matching score
        if score < best_score:
            best_score = score
            best_match = char
    return best_match

# Detect the characters on the license plate
def detect_plate_number(image_path, template_dir, verbose=False):
    # Step 1: Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = gaussian_blur(gray, kernel_size=5, verbose=verbose)

    # Step 2: Apply Sobel edge detection to find potential characters
    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edges = sobel_edge_detection(blurred, sobel_filter, verbose=verbose)

   # Step 3: Threshold the edge-detected image to get binary image
    edges_8bit = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(edges_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if verbose:
        plt.imshow(binary, cmap='gray')
        plt.title("Binary Image after Thresholding")
        plt.show()


    # Step 4: Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    if verbose:
        plt.imshow(morphed, cmap='gray')
        plt.title("Image after Morphological Operations")
        plt.show()

    # Step 5: Find contours (bounding boxes) of potential characters
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_characters = []

    # Step 6: Load templates for matching
    templates = load_templates(template_dir)

    # Step 7: Loop through contours, extract potential characters, and match them to templates
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 10 and w > 5:  # Adjust these thresholds to ignore small noise
            segment = morphed[y:y+h, x:x+w]

            # Match the extracted segment with the best template character
            matched_char = match_segment_with_template(segment, templates)
            if matched_char:
                detected_characters.append(matched_char)

            if verbose:
                plt.imshow(segment, cmap='gray')
                plt.title(f"Matched character: {matched_char}")
                plt.show()

    # Step 8: Print the detected plate number as text
    plate_number = ''.join(sorted(detected_characters))  # Sorted to arrange characters properly
    print(f"Detected Plate Number: {plate_number}")

    return plate_number

if __name__ == '__main__':
    # Define paths
    image_path = r"D:\ESCUELA\Laboratorio\Vision-ST\image.jpg"  # Path to the car plate image
    template_dir = r"D:\ESCUELA\Laboratorio\Vision-ST\templates"  # Path to template directory with digit images
    
    # Detect and print plate number
    detect_plate_number(image_path, template_dir, verbose=True)
