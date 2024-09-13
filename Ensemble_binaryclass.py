import os
import time
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
import shutil

# Load individual models for each orientation and maturity model
model_paths = {
    'top': '/mnt/c/Users/Mangga/Desktop/Raspi Ensemble August Files/Models/Grading/top/top_97_98_segmented.h5',
    'side': '/mnt/c/Users/Mangga/Desktop/Raspi Ensemble August Files/Models/Grading/side/side100_98_segmented.h5',
    'bottom': '/mnt/c/Users/Mangga/Desktop/Raspi Ensemble August Files/Models/Grading/bottom/bottom_98_96_Segmented.h5'
}

models = {orientation: load_model(path) for orientation, path in model_paths.items()}
class_names = ['Export', 'Non-export']
maturity_class_names = ['Mature', 'Immature']

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    image = cv2.resize(image, (416, 259))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def predict(model, img, class_labels):
    img_array = tf.expand_dims(img, 0)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_labels[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

def get_model_for_image(filename):
    if 'img1' in filename.lower():
        return 'top', models['top']
    elif 'img2' in filename.lower():
        return 'side', models['side']
    elif 'img3' in filename.lower():
        return 'bottom', models['bottom']
    return None, None

def process_mango_folder(mango_folder):
    csv_data = []
    sides = ['Side1', 'Side2', 'Side3', 'Side4']
    required_images = {'img1', 'img2', 'img3'}

    # Wait until all sides have the required images
    while True:
        all_sides_ready = True  # Assume all sides are ready initially

        for side in sides:
            side_path = os.path.join(mango_folder, side)
            if not os.path.exists(side_path):
                print(f"Waiting for {side_path} to be created...")
                all_sides_ready = False
                break

            files = {file.lower() for file in os.listdir(side_path)}
            if not any(file.startswith('img1') for file in files) or \
               not any(file.startswith('img2') for file in files) or \
               not any(file.startswith('img3') for file in files):
                print(f"Waiting for all required images in {side_path}...")
                all_sides_ready = False
                break

        if all_sides_ready:
            print("All required images are present in all side folders.")
            break

        time.sleep(1)

    # Process each side once all are ready
    for side in sides:
        side_path = os.path.join(mango_folder, side)
        files = sorted(os.listdir(side_path))
        row_data = {'Mango': os.path.basename(mango_folder) + side}

        for filename in files:
            image_path = os.path.join(side_path, filename)
            image = load_and_preprocess_image(image_path)
            if image is not None:
                orientation, model = get_model_for_image(filename)
                if orientation:
                    predicted_class, confidence = predict(model, image, class_names)
                    row_data[orientation] = predicted_class
            else:
                row_data[orientation] = 'Error'
        
        # Majority-based ensemble prediction
        predictions = [row_data.get('top'), row_data.get('side'), row_data.get('bottom')]
        if predictions.count('Export') > predictions.count('Non-export'):
            final_pred_class = 'Export'
        else:
            final_pred_class = 'Non-export'
        row_data['Final Prediction'] = final_pred_class

        # Maturity prediction
        maturity_predicted_class, _ = predict(models['top'], image, maturity_class_names)
        row_data['Maturity'] = maturity_predicted_class
        
        # Display results immediately
        print(f"{row_data['Mango']} - Final Prediction: {row_data['Final Prediction']}, Maturity: {row_data['Maturity']}")

        csv_data.append(row_data)

    return csv_data

# Base directory containing mango folders
base_folder_path = '/mnt/c/Users/Mangga/Desktop/ensemble binary test data/test data/'
processed_folder_path = '/mnt/c/Users/Mangga/Desktop/ensemble binary test data/processed_mangoes/'

# Create the processed folder if it doesn't exist
os.makedirs(processed_folder_path, exist_ok=True)

while True:
    mango_folders = sorted(os.listdir(base_folder_path))
    for mango_folder in mango_folders:
        mango_path = os.path.join(base_folder_path, mango_folder)
        try:
            if os.path.isdir(mango_path):
                csv_data = process_mango_folder(mango_path)

                # Save the predictions to a CSV file
                csv_df = pd.DataFrame(csv_data, columns=['Mango', 'side', 'bottom', 'top', 'Final Prediction', 'Maturity'])
                csv_save_path = os.path.join(base_folder_path, f'{mango_folder}_predictions_with_maturity.csv')
                csv_df.to_csv(csv_save_path, index=False)

                # Move processed mango folder to the processed folder
                shutil.move(mango_path, processed_folder_path)
        except Exception as e:
            print(f"Error processing {mango_folder}: {e}")

    print("Waiting for new mango folders...")
    time.sleep(5)  # Wait for 5 seconds before checking for new folders
