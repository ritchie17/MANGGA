import os
import shutil

# Define the source and destination directories
source_dir = '/mnt/c/Users/Mangga/Desktop/ensemble binary test data/'
destination_dir = '/mnt/c/Users/Mangga/Desktop/ensemble binary test data/arranged'

# Define the categories and image order
categories = ['Export', 'Reject']
image_order = ['Top', 'Side', 'Bottom']

# Create the destination folders if they don't exist
for category in categories:
    os.makedirs(os.path.join(destination_dir, category), exist_ok=True)

# Initialize a dictionary to store image paths by category and orientation
images = {'Export': {'Top': [], 'Side': [], 'Bottom': []},
          'Reject': {'Top': [], 'Side': [], 'Bottom': []}}

# Traverse the source directory and collect images
for orientation in image_order:
    orientation_path = os.path.join(source_dir, orientation)
    for category in categories:
        category_path = os.path.join(orientation_path, category)
        if os.path.exists(category_path):
            for file_name in os.listdir(category_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images[category][orientation].append(os.path.join(category_path, file_name))

# Get the minimum number of mangoes in any category to ensure even distribution
min_mangoes_export = min(len(images['Export']['Top']), len(images['Export']['Side']), len(images['Export']['Bottom']))
min_mangoes_reject = min(len(images['Reject']['Top']), len(images['Reject']['Side']), len(images['Reject']['Bottom']))

# Function to copy images for a category
def copy_images_for_category(category, min_mangoes):
    for i in range(min_mangoes):
        # Create a new folder for each mango
        mango_folder = os.path.join(destination_dir, category, f'mango_{i+1}')
        os.makedirs(mango_folder, exist_ok=True)

        for idx, orientation in enumerate(image_order):
            source_file = images[category][orientation][i]
            new_file_name = f'img{idx + 1}.jpg'
            destination_file = os.path.join(mango_folder, new_file_name)
            shutil.copy(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")

# Copy images for 'Export' and 'Reject' categories
copy_images_for_category('Export', min_mangoes_export)
copy_images_for_category('Reject', min_mangoes_reject)

print("All images have been successfully copied and organized.")
