import os

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
images = []
classes = []
train_img = None
test_img = None
train_classes = None
test_classes = None
datagen = None


def load_digits():
    global images
    global classes
    global train_img
    global test_img
    global train_classes
    global test_classes

    data = os.listdir("./digits")
    for folder_name in data:
        folder = os.listdir("./digits/" + folder_name)
        for picture in folder:
            image = cv2.imread("./digits/" + folder_name + "/" + picture)
            image = cv2.resize(image, (32, 32))
            images.append(image)
            classes.append(int(folder_name) - 1)
    images = np.array(images)
    classes = np.array(classes)
    train_img, test_img, train_classes, test_classes = train_test_split(images, classes, test_size=0.2)


def preprocess_data():
    global train_img
    global test_img
    global datagen

    temp_img = []
    for img in train_img:
        temp_img.append(prep(img))
    train_img = np.array(temp_img)

    temp_img = []
    for img in test_img:
        temp_img.append(prep(img))
    test_img = np.array(temp_img)

    train_img = train_img.reshape(train_img.shape[0], train_img.shape[1], train_img.shape[2], 1)
    test_img = test_img.reshape(test_img.shape[0], test_img.shape[1], test_img.shape[2], 1)
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                                 rotation_range=10)
    datagen.fit(train_img)


def create_matrix_classes():
    global train_classes
    global test_classes

    number_of_classes = 9  # 1,2,3,4,5,6,7,8,9
    train_classes = to_categorical(train_classes, number_of_classes)
    test_classes = to_categorical(test_classes, number_of_classes)


def prep(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def create_model(file, epochs):
    model = Sequential()

    model.add((Conv2D(32, (5, 5), input_shape=(32, 32, 1), padding="same", activation="relu")))
    model.add((Conv2D(32, (5, 5), padding="same", activation="relu")))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add((Conv2D(16, (3, 3), padding="same", activation="relu")))
    model.add((Conv2D(16, (3, 3), padding="same", activation="relu")))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(300, activation="relu"))

    model.add(Dense(9, activation="softmax"))

    model.summary()

    # Compiling the model
    optimizer = Adam(learning_rate=0.002)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    global train_img
    global train_classes

    history = model.fit(datagen.flow(train_img, train_classes, batch_size=32),
                        epochs=epochs, steps_per_epoch=200,
                        validation_data=(test_img, test_classes))

    model.save(file)


if __name__ == '__main__':
    load_digits()
    preprocess_data()
    create_matrix_classes()
    create_model("digits_model.h5", 30)
