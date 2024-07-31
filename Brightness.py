from PIL import Image
import numpy as np
import os
import shutil
import pandas as pd

def calculate_brightness(image):
    # Convert image data to numpy array
    image_data = np.array(image)
    # Create a mask for pixels where all RGB values are greater than 0
    mask = np.all(image_data > 0, axis=-1)
    # Calculate the weighted sum of the R, G, and B values for non-black pixels
    if np.any(mask):
        # Select only non-black pixels
        non_black_pixels = image_data[mask]
        # Calculate weighted brightness
        weights = np.array([0.299, 0.587, 0.114])
        brightness = np.dot(non_black_pixels, weights).mean()
        return brightness
    else:
        return 0  # In case the image is entirely black

def categorize_brightness(brightness):
    # Categorize brightness based on defined thresholds (Adjust this according to your liking or look through the first results and manually look through the value and adjust it)
    if brightness <= 101.86:
        return 1  # Too Dark
    elif brightness <= 110.25:
        return 2  # Less Bright
    elif brightness <= 117.34:
        return 3  # Bright
    else:
        return 4  # Too Bright

def sort_images_by_brightness(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    brightness_categories = {1: [], 2: [], 3: [], 4: []}
    image_details = []

    # Check and load images
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            try:
                with Image.open(img_path) as img:
                    brightness = calculate_brightness(img)
                    category = categorize_brightness(brightness)
                    brightness_categories[category].append(img_path)
                    output_path = os.path.join(output_folder, f'category_{category}')
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    shutil.copy(img_path, output_path)
                    
                    # Append details to DataFrame
                    image_details.append({'File Name': filename, 'Brightness': brightness})
            except Exception as e:
                print(f"Failed to process {filename}. Error: {e}")

    # Save DataFrame to Excel
    brightness_df = pd.DataFrame(image_details)
    brightness_df.to_excel(os.path.join(output_folder, 'brightness_values.xlsx'), index=False)
    return brightness_categories

# Specify your input and output directory paths
input_folder = '/mnt/d/AI Internship Repo/For Kenz Dataset Sort/Side/Reject'  # Change this to your input directory path
output_folder = '/mnt/d/AI Internship Repo/For Kenz Dataset Sort/Side Brightness/Reject'  # Change this to your output directory path

sorted_images = sort_images_by_brightness(input_folder, output_folder)
for category, images in sorted_images.items():
    category_name = {1: "Too Dark", 2: "Less Bright", 3: "Bright", 4: "Too Bright"}[category]
    print(f'{category_name} has {len(images)} images.')
