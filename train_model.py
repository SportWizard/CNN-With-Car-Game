import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import img_to_array, to_categorical
from sklearn.model_selection import train_test_split


def construct_data(height, width):
    dataset_dir = "outputs"

    # Training data
    data = []
    labels = []

    # Define mapping for the labels
    label_map = {"forward": 0, "left": 1, "right": 2}

    for label in label_map: # Loop through the keys
        dir_path = os.path.join(dataset_dir, label)

        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            img = cv2.imread(file_path)

            img = cv2.resize(img, (height, width)) # Resize image due to CNN expecting a fixed-size input
            img = img_to_array(img) # Convert the image into an array

            img = img / 255.0 # Normalize pixel values from [0, 255] to [0, 1]

            data.append(img)
            labels.append(label_map[label])

    return np.array(data), np.array(labels) # Convert lists to NumPy arrays because NumPy arrays are like vector

def build(height, width, channels, classes):
    input_shape = (height, width, channels)

    model = models.Sequential()

    # Data augmentation
    layers.RandomContrast(factor=0.5, seed=32)

    # First convolutional layer
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second convolutional layer
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Third convolutional layer
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Dense layer (hidden layer)
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten()) # Convert multi-dimensional tensor into a one-dimensional vector
    model.add(layers.Dense(512))
    model.add(layers.Activation("relu"))

    # Output layer
    model.add(layers.Dense(classes))
    model.add(layers.Activation("softmax"))

    return model

def main():
    height = 64 # Height of the image
    width = 64 # Width of the image
    channels = 3 # Number of channel (RGB has 3 channels)

    classes = 3 # Number of outputs

    print("Constructing data...")
    data, labels = construct_data(height, width)
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=204, test_size=0.20)

    # One-Hot Encoding (e.g. 0 becomes [1, 0, 0], 1 becomes [0, 1, 0], 2 becomes [0, 0, 1]). Used for multi-class classification
    y_train = to_categorical(y_train, num_classes=classes)
    y_test = to_categorical(y_test, num_classes=classes)

    epochs = 100
    batch_size = 32

    # Initialize the model
    print("Compiling model...")
    model = build(height, width, channels, classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    early_stopping = EarlyStopping(
        min_delta=0.001, # Minimium amount of change to count as an improvement
        patience=20, # How many epochs to wait before stopping
        restore_best_weights=True
    )

    # epochs: the number of times to use all the training data
    # batch_size: the subset of the training data used for each iteration
    # verbose: a mode or option that provides detailed output or information about the processes being executed
    print("Training network...")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

    # save the model to disk
    print("Saving model...")
    model.save("model.keras")

if __name__ == "__main__":
    main()