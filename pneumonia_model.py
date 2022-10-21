from keras.preprocessing import image
from matplotlib.pyplot import imshow
from typing import Any
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.utils import load_img
from keras.utils.image_utils import img_to_array

load_params = { 'target_size': (64, 64), 'batch_size': 64, 'class_mode': 'categorical' }

# Processing the trainset
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train = train_datagen.flow_from_directory('chest_xray/train', **load_params)

# Processing the testset
test_datagen = ImageDataGenerator(rescale=1./255)
test = test_datagen.flow_from_directory('chest_xray/test', **load_params)

# Initialising the CNN model
model = Sequential([Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),  # Convolutional feature mapping and rectifier activation --> correcting linearity
                    MaxPool2D(pool_size=2, strides=2), # Pooling --> reducing features to avoid overfitting
                    Conv2D(filters=32, kernel_size=3, activation='relu'), # Adding a second convolutional layer
                    Flatten(),  # Flattening
                    Dense(units=128, activation='sigmoid'),  # Full Connection
                    Dense(units=2, activation='softmax')])  # Output Layer

# Compiling the CNN model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the CNN model
model.fit(x=train, validation_data=test, epochs=10)


# Save model to disk
model.save("cnn.h5")

examples = ['chest_xray/test/NORMAL/NORMAL-745902-0001.jpeg', 'chest_xray/test/PNEUMONIA/BACTERIA-227418-0001.jpeg', 'chest_xray/train/PNEUMONIA/BACTERIA-7422-0001.jpeg']

for example in examples:
    ex = load_img(example, target_size=(64, 64))
    ex = np.expand_dims(img_to_array(ex), axis=0)

    prediction = model.predict(ex)
    print(prediction)
