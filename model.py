import os
import sys
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D
import csv
import cv2
import numpy as np

def read_data(path):
    lines = []
    images = []
    measurements = []
    with open(os.path.join(path,'driving_log.csv')) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    for line in lines:
        source_path = line[0]
        filename = source_path.split("/")[-1]
        current_path = os.path.join(path,'IMG/' + filename)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    return lines, np.array(images), np.array(measurements)


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
    return model

def train_model(model, x_train, y_train):
    history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])

    # Save model
    model.save('model.h5')

if __name__ == "__main__":
    
    lines, X_train, y_train = read_data("./data/")
    model = create_model()

    model.compile(optimizer='adam', loss='mse')
    
    train_model(model, X_train, y_train)