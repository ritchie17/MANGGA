import os
import shutil
from PIL import Image
import pandas as pd
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory containing images
image_dir = '/mnt/d/Dataset August/Original Extracted Orientation/Side/Reject/'
output_excel = '/mnt/d/Dataset August/Original Extracted Orientation/Side/mango_classification.xlsx'  # Updated with filename

# Make directories for each category
output_dirs = {
    'Rounded': '/mnt/d/Dataset August/Original Extracted Orientation/Side/Rounded',
    'Slightly elongated': '/mnt/d/Dataset August/Original Extracted Orientation/Side/Slightly Elongated',
    'Highly elongated': '/mnt/d/Dataset August/Original Extracted Orientation/Side/Highly Elongated'
}
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)
    logging.info(f"Directory {dir_path} created.")

# Function to calculate the width-to-height ratio
def calculate_ratio(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('L')
            binary_image = img.point(lambda p: p > 50 and 255)
            bbox = binary_image.getbbox()
            if bbox is None:
                logging.warning(f"No bounding box found for {image_path}.")
                return None
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            ratio = width / height
            return ratio
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

# Categorize based on the ratio
def categorize_ratio(ratio):
    if ratio <= 1.5:
        return 'Rounded'
    elif ratio <= 2.0:
        return 'Slightly elongated'
    else:
        return 'Highly elongated'

# Process all images and copy them to the respective category folder
data = []
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(image_dir, filename)
        ratio = calculate_ratio(filepath)
        if ratio is None:
            logging.info(f"Skipping {filename} due to no detectable ratio.")
            continue
        category = categorize_ratio(ratio)
        logging.info(f"File {filename} has a ratio of {ratio:.2f} and is categorized as {category}.")
        data.append({'Filename': filename, 'Value': ratio, 'Category': category})
        shutil.copy(filepath, os.path.join(output_dirs[category], filename))
        logging.info(f"File {filename} copied to {output_dirs[category]}.")

# Create a DataFrame and save to Excel
df = pd.DataFrame(data)
df.to_excel(output_excel, index=False)
logging.info(f"Analysis complete. Data saved to {output_excel}")
