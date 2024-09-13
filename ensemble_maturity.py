import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load individual models for each orientation
model_paths = {
    'top': '/mnt/f/Test Set Ensemble/models/top.h5',  # Top orientation model
    'side': '/mnt/f/Test Set Ensemble/models/side.h5',  # Side orientation model
    'bottom': '/mnt/f/Test Set Ensemble/models/bottom.h5',  # Bottom orientation model
    'maturity': '/mnt/f/Test Set Ensemble/models/maturity.h5'  # Maturity model
}
models = {orientation: load_model(path) for orientation, path in model_paths.items()}
class_names = ['export', 'local', 'reject']

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    image = cv2.resize(image, (461, 259))  # Correct width and height to match model input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def predict(model, img):
    img_array = tf.expand_dims(img, 0)  # Create a batch
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence, np.argmax(predictions[0]), predictions[0] 

def get_model_for_image(filename):
    if 'img1' in filename:
        return 'top', models.get('top')
    elif 'img2' in filename:
        return 'side', models.get('side')
    elif 'img3' in filename:
        return 'bottom', models.get('bottom')
    return None, None

def predict_maturity(image):
    maturity_model = models['maturity']
    img_array = tf.expand_dims(image, 0)
    predictions = maturity_model.predict(img_array, verbose=0)
    maturity_label = 'mature' if np.argmax(predictions[0]) == 1 else 'immature'
    confidence = round(100 * np.max(predictions[0]), 2)
    return maturity_label, confidence

print(models['top'].summary())
print(models['side'].summary())
print(models['bottom'].summary())
print(models['maturity'].summary())

base_folder_path = '/mnt/f/Test Set Ensemble/'
class_labels = ['export', 'local', 'reject']

all_true_labels = []
all_predicted_labels = []

# Process each class label directory
for class_label in class_labels:
    class_path = os.path.join(base_folder_path, class_label)
    for batch_folder in os.listdir(class_path):
        batch_path = os.path.join(class_path, batch_folder)
        files = sorted(os.listdir(batch_path))
        if len(files) == 3:  # Ensure exactly three images are present
            plt.figure(figsize=(19.2, 10.8))
            plt.suptitle(f"Processing folder: {batch_folder}", fontsize=16)
            ensemble_inputs = []
            individual_predictions = []
            prediction_probs = []

            for i, filename in enumerate(files):
                image_path = os.path.join(batch_path, filename)
                image = load_and_preprocess_image(image_path)
                if image is not None:
                    orientation, model = get_model_for_image(filename)
                    predicted_class, confidence, pred_label, pred_prob = predict(model, image)
                    ensemble_inputs.append((image, model))
                    individual_predictions.append(pred_label)
                    prediction_probs.append(pred_prob)
                    ax = plt.subplot(1, 5, i + 1)
                    plt.imshow(image.astype("uint8"), aspect='0.56182212581')
                    plt.title(f"{filename}\nPredicted: {predicted_class}\nConfidence: {confidence}%", color='green' if predicted_class == class_label else 'red')
                    plt.axis("off")

            # Rule-based ensemble prediction with maturity integration
            if len(ensemble_inputs) == 3:
                maturity_label, maturity_confidence = predict_maturity(ensemble_inputs[0][0])  # Using the first image for maturity prediction

                if all(pred == 0 for pred in individual_predictions):  # 0 corresponds to 'export'
                    if maturity_label == 'mature':
                        final_pred_class = 'export'
                    else:
                        final_pred_class = 'reject'
                    final_confidence = 100.0  # 100% confidence for unanimous 'export'
                elif any(pred == 2 for pred in individual_predictions):  # 2 corresponds to 'reject'
                    final_pred_class = 'reject'
                    final_confidence = 100.0  # 100% confidence for any 'reject'
                else:
                    average_probs = np.mean(prediction_probs, axis=0)
                    final_pred_class = 'local'  # Can only be 'local' if not all 'export' and none 'reject'
                    final_confidence = round(100 * average_probs[1], 2)  # Confidence for 'local'

                all_true_labels.append(class_names.index(class_label))  # Append true label for ensemble prediction
                all_predicted_labels.append(class_names.index(final_pred_class))  # Append ensemble prediction
                ax = plt.subplot(1, 5, 4)
                plt.imshow(np.hstack([img for img, _ in ensemble_inputs]).astype("uint8"), aspect='0.56182212581')
                plt.title(f"Ensemble Prediction: {final_pred_class}\nMaturity: {maturity_label}", color='green' if final_pred_class == class_label else 'red')
                plt.axis("off")
                ax = plt.subplot(1, 5, 5)
                plt.imshow(ensemble_inputs[0][0].astype("uint8"), aspect='0.56182212581')
                plt.title(f"Maturity: {maturity_label}\nConfidence: {maturity_confidence}%")
                plt.axis("off")
                plt.show()

# Display the confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels, labels=range(len(class_names)))
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Overall Confusion Matrix")
plt.show()
