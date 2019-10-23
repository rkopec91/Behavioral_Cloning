import os
import sys
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D
import csv
import cv2
import numpy as np
from tensorflow.python.client import device_lib
import tensorflow as tf
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def read_lines():
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]

                    filename = source_path.split("/")[-1]

                    current_path = os.path.join('./data/IMG/', filename)

                    image = cv2.imread(current_path)

                    images.append(image)
                    images.append(np.fliplr(image))

                    measurement = float(batch_sample[3])
                    if i == 1:
                        angles.append(measurement+0.2)
                        angles.append(-measurement-0.2)
                    elif i == 2:
                        angles.append(measurement-0.2)
                        angles.append(-measurement+0.2)
                    else:
                        angles.append(measurement)
                        angles.append(-measurement)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, samples):

    train_samples, validation_samples = train_test_split(samples,test_size=0.15)

    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

    # Save model
    model.save('model.h5')

    model.summary()

if __name__ == "__main__":
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.tensorflow_backend.set_session(sess)
    samples = read_lines()
    model = create_model()

    train_model(model, samples)