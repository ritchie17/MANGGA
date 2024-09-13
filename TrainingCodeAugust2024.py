import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs except errors

# Suppress CPU and GPU warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

##############################################################
# Configuration and Constants
batch_size = 16
epochs = 500
img_width = 416
img_height = 259
default_image_size = (img_height, img_width)  # Ensure dimensions are consistent (height, width)
data_dir = '/mnt/d/Dataset August/Grading/Export VS Non-export/Side'
channels = 3
AUTOTUNE = tf.data.AUTOTUNE
train_split = 0.75
val_split = 0.15
test_split = 0.10
##############################################################
# Prompt user for training type and orientation
data_name = input("Enter the name of the data being trained: ")
training_type = input("Enter classification type (maturity or grading): ").lower()
orientation = input("Enter the orientation (top, side, or bottom): ").lower()

##############################################################
# Create a directory to save outputs based on training type and orientation
def create_output_dir():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(training_type, orientation, f"Training_run_{current_time}_{data_name}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, current_time

output_dir, current_time = create_output_dir()

##############################################################

##############################################################
# Function to save figures
def save_figure(fig, filename):
    fig.savefig(os.path.join(output_dir, filename))

##############################################################
# Function to split dataset into training, validation, and test sets
def split_dataset(dataset, train_split, val_split, test_split):
    # Calculate sizes for each split
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Perform the actual split
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size).take(test_size)

    return train_ds, val_ds, test_ds, train_size, val_size, test_size

##############################################################
# Function to save test dataset images
def save_test_images(test_ds, output_dir, dataset_class_names):
    # Create a directory to save test images
    test_images_dir = os.path.join(output_dir, 'test_images')
    os.makedirs(test_images_dir, exist_ok=True)

    # Iterate through the test dataset and save images
    for batch_idx, (images, labels) in enumerate(test_ds):
        for i in range(len(images)):
            image = images[i].numpy()
            label = labels[i].numpy()
            class_name = dataset_class_names[label]
            
            # Normalize the image to range [0, 1]
            image = (image - image.min()) / (image.max() - image.min())
            
            # Create a directory for each class
            class_dir = os.path.join(test_images_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Save the image
            image_path = os.path.join(class_dir, f'image_{batch_idx * batch_size + i}.png')
            plt.imsave(image_path, image)

##############################################################
# Load dataset and split into train, validation, test
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=default_image_size,
    batch_size=batch_size,  # Use the desired batch size here
    shuffle=True,
    seed=42,
    label_mode='int'
)

train_ds, val_ds, test_ds, train_size, val_size, test_size = split_dataset(dataset, train_split, val_split, test_split)

# Cache the test dataset to ensure consistency
test_ds = test_ds.cache()

# Save test dataset images
save_test_images(test_ds, output_dir, dataset.class_names)

train_distribution = train_size * batch_size
validation_distribution = val_size * batch_size
test_distribution = test_size * batch_size
total_dataset = train_distribution + validation_distribution +test_distribution

print(f"Training dataset size: {train_distribution}")
print(f"Validation dataset size: {validation_distribution}")
print(f"Test dataset size: {test_distribution}")

# Prefetch datasets
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

##############################################################
# Define data preprocessing layers
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(img_height, img_width),  # Ensure dimensions are (height, width)
    layers.Rescaling(1./255),
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

##############################################################
# Build the model
model = models.Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, channels)),  # Ensure input shape matches (height, width, channels)
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(len(dataset.class_names), activation='softmax'),
])

model.build((None, img_height, img_width, channels))
model.summary()

##############################################################
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

##############################################################
# Define callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=20, min_lr=0.000001)
temp_model_path = os.path.join(output_dir, 'temp_model.h5')
model_checkpoint = ModelCheckpoint(filepath=temp_model_path, monitor="val_accuracy",
                                   save_best_only=True, mode="auto", save_freq="epoch")
csv_logger = CSVLogger(os.path.join(output_dir, 'training_log.csv'))

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[reduce_lr,model_checkpoint, csv_logger]  # early_stopping, 
)

##############################################################
# Evaluate the best model on validation and test sets
best_model = models.load_model(temp_model_path)
best_scores_val = best_model.evaluate(val_ds)
best_scores_test = best_model.evaluate(test_ds)

print(f"Validation Accuracy: {best_scores_val[1]}")
print(f"Test Accuracy: {best_scores_test[1]}")

# Rename the model file
validation_accuracy = int(best_scores_val[1] * 100)
test_accuracy = int(best_scores_test[1] * 100)
new_model_name = f"{training_type}_{current_time}_{validation_accuracy}_{test_accuracy}_{data_name}.h5"
new_model_path = os.path.join(output_dir, new_model_name)
os.rename(temp_model_path, new_model_path)

##############################################################
# Plot and save training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(epochs_range, acc, label='Training Accuracy')
ax[0].plot(epochs_range, val_acc, label='Validation Accuracy')
ax[0].legend(loc='lower right')
ax[0].set_title('Training and Validation Accuracy')

ax[1].plot(epochs_range, loss, label='Training Loss')
ax[1].plot(epochs_range, val_loss, label='Validation Loss')
ax[1].legend(loc='upper right')
ax[1].set_title('Training and Validation Loss')

save_figure(fig, 'training_validation_accuracy_loss.png')
plt.show()

##############################################################
# Function to calculate confusion matrix
def calculate_confusion_matrix(ds, best_model):
    true_labels = []
    predicted_labels = []

    for images, labels in ds:
        true_labels.extend(labels.numpy())
        predictions = best_model.predict(images)
        predicted_labels.extend(np.argmax(predictions, axis=1))

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    return conf_matrix

# Calculate confusion matrices
conf_matrix_train = calculate_confusion_matrix(train_ds, best_model)
conf_matrix_val = calculate_confusion_matrix(val_ds, best_model)
conf_matrix_test = calculate_confusion_matrix(test_ds, best_model)

##############################################################
# Display and save confusion matrices
disp_train = ConfusionMatrixDisplay(conf_matrix_train, display_labels=dataset.class_names)
fig_train, ax_train = plt.subplots()
disp_train.plot(cmap=plt.cm.summer_r, values_format="d", ax=ax_train)
ax_train.set_title("Training Dataset Confusion Matrix")
save_figure(fig_train, 'confusion_matrix_train.png')
plt.show()

disp_val = ConfusionMatrixDisplay(conf_matrix_val, display_labels=dataset.class_names)
fig_val, ax_val = plt.subplots()
disp_val.plot(cmap=plt.cm.summer_r, values_format="d", ax=ax_val)
ax_val.set_title("Validation Dataset Confusion Matrix")
save_figure(fig_val, 'confusion_matrix_val.png')
plt.show()

disp_test = ConfusionMatrixDisplay(conf_matrix_test, display_labels=dataset.class_names)
fig_test, ax_test = plt.subplots()
disp_test.plot(cmap=plt.cm.summer_r, values_format="d", ax=ax_test)
ax_test.set_title("Test Dataset Confusion Matrix")
save_figure(fig_test, 'confusion_matrix_test.png')
plt.show()

# Function to save metrics and CNN layers information to a text file
##########################################################
train_percentsplit = train_split *100
val_percentsplit = val_split *100
test_percentsplit = test_split *100

# Function to save metrics and CNN layers information to a text file
def save_metrics_info(output_dir, data_dir, batch_size, epochs, starting_lr,
                      total_datasets, train_size, val_size, test_size,
                      validation_accuracy, test_accuracy, validation_loss, model_summary):
    metrics_info = f"Dataset Directory: {data_dir}\n" \
                   f"Batch Size: {batch_size}\n" \
                   f"Epochs: {epochs}\n" \
                   f"Starting Learning Rate: {starting_lr}\n" \
                   f"Total Number of Datasets: {total_dataset}\n" \
                   f"Train_Validation_Test_Ratio: {train_percentsplit}_{val_percentsplit}_{test_percentsplit}\n"\
                   f"Training Dataset Size: {train_distribution}\n" \
                   f"Validation Dataset Size: {validation_distribution}\n" \
                   f"Test Dataset Size: {test_distribution}\n" \
                   f"Validation Accuracy: {validation_accuracy}\n" \
                   f"Test Accuracy: {test_accuracy}\n" \
                   f"Validation Loss for Best Model: {validation_loss}\n\n" \
                   f"Model Summary:\n{model_summary}"\

    metrics_file = os.path.join(output_dir, 'metrics_info.txt')
    with open(metrics_file, 'w') as f:
        f.write(metrics_info)

# Retrieve and format the model summary
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
model_summary = '\n'.join(model_summary)

# Call the function to save metrics information including the model summary
save_metrics_info(output_dir, data_dir, batch_size, epochs, 0.01,
                  train_size + val_size + test_size, train_size, val_size, test_size,
                  best_scores_val[1], best_scores_test[1], val_loss[-1], model_summary)
