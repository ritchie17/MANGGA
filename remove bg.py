from PIL import Image
from rembg import remove
import os

# Specify the root directory where your images are located
root_directory = "D:\AI Internship Repo\Renamed & FS Categorized Revised"


# Recursively process images in subdirectories
for foldername, subfolders, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):    
            input_path = os.path.join(foldername, filename)
            output_path = os.path.join(foldername, f"{filename}")

            # Processing the image
            input_image = Image.open(input_path)
            output_image = remove(input_image)

            # Create a new image with black background
            black_background = Image.new("RGB", input_image.size, color="black")
            black_background.paste(output_image, (0, 0), output_image)

            # Save the isolated mango with black background
            black_background.save(output_path)

            print(f"Processed: {input_path} -> {output_path}")
