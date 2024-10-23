import cv2
import numpy as np
import os

def create_templates(output_dir, font=cv2.FONT_HERSHEY_SIMPLEX, image_size=(28, 28)):
    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List of characters to create templates for (you can add more if needed)
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for char in characters:
        # Create a blank image
        template = np.ones(image_size, dtype=np.uint8) * 255  # White background
        
        # Put the character in the middle of the image
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(char, font, font_scale, thickness)[0]
        text_x = (image_size[1] - text_size[0]) // 2
        text_y = (image_size[0] + text_size[1]) // 2
        cv2.putText(template, char, (text_x, text_y), font, font_scale, (0,), thickness, lineType=cv2.LINE_AA)

        # Save the image with the character as the filename
        output_path = os.path.join(output_dir, f"{char}.png")
        cv2.imwrite(output_path, template)

        print(f"Saved template for character '{char}' at {output_path}")

if __name__ == "__main__":
    # Define the output directory where templates will be saved
    output_dir = r"D:\ESCUELA\Laboratorio\Vision-ST\templates"  # Change this to your desired path
    
    # Generate the templates
    create_templates(output_dir)
