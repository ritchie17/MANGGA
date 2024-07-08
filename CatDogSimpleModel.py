import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from google.colab import drive
drive.mount('/content/gdrive')

################################
batch_size = 16
img_width = 64
img_height = 64
image_size = (img_width,img_height)
epoch = 100
train_dir = '/content/gdrive/MyDrive/Interns_Training_AI/Cat_Dog/Train/'
validation_dir = '/content/gdrive/MyDrive/Interns_Training_AI/Cat_Dog/Validation/'

##################################################

train_generator = ImageDataGenerator(rescale= 1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
train_set = train_generator.flow_from_directory(train_dir,
                                                target_size = (image_size),
                                                batch_size =batch_size ,
                                                class_mode ='categorical')

validation_generator = ImageDataGenerator(rescale = 1./255) #no preprocessing only normalize

validation_set = validation_generator.flow_from_directory(validation_dir,
                                                          target_size= (image_size),
                                                          batch_size = batch_size,
                                                          class_mode = 'categorical')


#########################################

model = Sequential()
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape = [img_width,img_height,3]))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape = [img_width,img_height,3]))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape = [img_width,img_height,3]))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))

######################
print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

########################

tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
from IPython.display import Image
Image(filename='model.png')


history = model.fit(x = train_set, validation_data = validation_set, epochs = epoch)