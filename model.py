import os
import sys
from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D
import csv
import cv2
import numpy as np
from tensorflow.python.client import device_lib

def read_data(path):
    lines = []
    images = []
    measurements = []
    with open(os.path.join(path,'driving_log.csv')) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    for line in lines:
        for i in range(3):
            source_path = line[i]

            filename = source_path.split("/")[-1]

            current_path = os.path.join(path,'IMG/' + filename)

            image = cv2.imread(current_path)

            images.append(image)
            images.append(np.fliplr(image))

            measurement = float(line[3])
            if i == 1:
                measurements.append(measurement+0.2)
                measurements.append(-measurement-0.2)
            elif i == 2:
                measurements.append(measurement-0.2)
                measurements.append(-measurement+0.2)
            else:
                measurements.append(measurement)
                measurements.append(-measurement)

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
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train):
    
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)

    # Save model
    model.save('model.h5')

if __name__ == "__main__":
    print(device_lib.list_local_devices())
    K.tensorflow_backend._get_available_gpus()

    lines, X_train, y_train = read_data("./data/")
    model = create_model()

    train_model(model, X_train, y_train)