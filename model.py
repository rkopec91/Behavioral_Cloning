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
    """
    Output:
        samples: array of each line within the csv file.
    Reads each line of the driving_log.csv file.
    """
    samples = []
    # Opens the csv file.
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        # Loops through each line in the csv file.
        for line in reader:
            # Append each line to the samples array
            samples.append(line)
    return samples


def generator(samples, batch_size=32):
    """
    Inputs:
        samples: Array of lines from the csv file
        batch_size: the amount of samples in each batch that is yeilded.
    """
    # Amount of samples
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # Shuffles all the samples
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # pulls the batch samples
            batch_samples = samples[offset:offset+batch_size]
            # Creates an empty array of images and angles(for steering)
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Loops through the three cameras on the dash of the car.
                for i in range(3):
                    # Gets the image paths
                    source_path = batch_sample[i]

                    filename = source_path.split("/")[-1]

                    current_path = os.path.join('./data/IMG/', filename)
                    # Reads the images
                    image = cv2.imread(current_path)
                    # Appends the image to the array of images and flips the image and appends again
                    images.append(image)
                    images.append(np.fliplr(image))
                    # gets the angle measurement
                    measurement = float(batch_sample[3])
                    if i == 1:
                        # If this is from the left camera, manipulates the measurements by adding and subtracting 0.2
                        angles.append(measurement+0.2)
                        angles.append(-measurement-0.2)
                    elif i == 2:
                        # If this is from the right camera, manipulates the measurements by subtracting and adding 0.2
                        angles.append(measurement-0.2)
                        angles.append(-measurement+0.2)
                    else:
                        # Appends measurements for the centered camera
                        angles.append(measurement)
                        angles.append(-measurement)

            X_train = np.array(images)
            y_train = np.array(angles)
            # yield the suffled y and x train
            yield sklearn.utils.shuffle(X_train, y_train)


def create_model():
    """
    Create's the model.
    output:
        model: The neural network model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    # Crops the image to only see the road
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    # Convolutional layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    #Drop out
    model.add(Dropout(0.8))
    # Flatten
    model.add(Flatten())
    # Dense
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, samples):
    """
    input:
        model:  The neural network model
        samples:  The samples generated from read_lines
    """
    train_samples, validation_samples = train_test_split(samples,test_size=0.15)

    # Generator for the train and validation samples
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    # Fits the model to the generator samples
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

    # Save model
    model.save('model.h5')
    # Prints summary
    model.summary()

if __name__ == "__main__":
    # Set the gpu options
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.tensorflow_backend.set_session(sess)

    # Create the samples and model
    samples = read_lines()
    model = create_model()
    # Train the model
    train_model(model, samples)