import os
import time
import keras.models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PIL import Image, ImageOps
import cv2
from imutils import contours
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import RMSprop, Adam

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
    data = os.listdir("./digits")
    for folder_name in data:
        folder = os.listdir("./digits/" + folder_name)
        for picture in folder:
            image = cv2.imread("./digits/" + folder_name + "/" + picture)
            image = cv2.resize(image, (32, 32))
            images.append(image)
            classes.append(int(folder_name))
    images = np.array(images)
    classes = np.array(classes)


def split_data():
    global train_img
    global test_img
    global train_classes
    global test_classes
    train_img, test_img, train_classes, test_classes = train_test_split(images, classes, test_size=0.2)


def prep(img):
    # img = ImageOps.grayscale(img)  # preprocessing - making image grayscale
    # img = ImageOps.equalize(img, mask=None)  # preprocessing - Histogram equalization to enhance contrast
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # making image grayscale
    img = cv2.equalizeHist(img)  # Histogram equalization to enhance contrast
    img = img / 255  # normalizing
    return img


def preprocess_data():
    global train_img
    global test_img
    global datagen
    train_img = np.array(list(map(prep, train_img)))
    test_img = np.array(list(map(prep, test_img)))

    # Reshaping the images
    train_img = train_img.reshape(train_img.shape[0], train_img.shape[1], train_img.shape[2], 1)
    test_img = test_img.reshape(test_img.shape[0], test_img.shape[1], test_img.shape[2], 1)
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                                 rotation_range=10)
    datagen.fit(train_img)


def create_matrix_classes():
    global train_classes
    global test_classes
    number_of_classes = 10  # 0,1,2,3,4,5,6,7,8,9
    train_classes = to_categorical(train_classes, number_of_classes)
    test_classes = to_categorical(test_classes, number_of_classes)


def preprocess_board(image):
    image = cv2.resize(image, (450, 450))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 6)
    # blur = cv2.bilateralFilter(gray,9,75,75)
    threshold_img = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return threshold_img, image


def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new


def splitcells(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


def do_crop(image, cnt1, cnt2):
    biggest, maxArea = main_outline(cnt1)
    if biggest.size != 0:
        biggest = reframe(biggest)
        cv2.drawContours(cnt2, biggest, -1, (0,255,0), 10)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imagewrap = cv2.warpPerspective(image, matrix, (450, 450))
        imagewrap = cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("aaa", imagewrap)
        return imagewrap
    return None

def find_contours(image, threshold):
    image1 = image.copy()
    image2 = image.copy()
    cnt, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image1, cnt, -1, (0, 255, 0), 3)
    # cv2.imshow("naslov", image1)
    # cv2.waitKey(1000)
    return image, cnt, image2


# def crop_sudoku_board(image):
#     # Load image, grayscale, adaptive threshold
#     result = image.copy()
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
#
#     # Fill rectangular contours
#     cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     for c in cnts:
#         cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)
#
#     # Morph open
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
#
#     # Draw rectangles
#     cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     cropped_image = None
#     for c in cnts:
#         x, y, w, h = cv2.boundingRect(c)
#         cropped_image = image[y:y + h, x:x + w]
#     return cropped_image


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


# def create_board_matrix(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)
#
#     # Filter out all numbers and noise to isolate only boxes
#     cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     for c in cnts:
#         area = cv2.contourArea(c)
#         if area < 1000:
#             cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
#
#     # Fix horizontal and vertical lines
#     vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
#     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)
#
#     # Sort by top to bottom and each row by left to right
#     invert = 255 - thresh
#     cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
#
#     sudoku_rows = []
#     row = []
#     for (i, c) in enumerate(cnts, 1):
#         area = cv2.contourArea(c)
#         if area < 50000:
#             row.append(c)
#             if i % 9 == 0:
#                 (cnts, _) = contours.sort_contours(row, method="left-to-right")
#                 sudoku_rows.append(cnts)
#                 row = []
#
#     # Iterate through each box
#     model = keras.models.load_model("digits_model.h5")
#     for row in sudoku_rows:
#         for c in row:
#             mask = np.zeros(image.shape, dtype=np.uint8)
#             cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
#             result = cv2.bitwise_and(image, mask)
#             gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#             coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
#             x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
#             square = result[y:y + h, x:x + w]
#             square = cv2.resize(square, (32, 32))
#             # square = apply_brightness_contrast(square, 0, 64)
#
#             square2 = Image.fromarray(square).convert('L')
#             square2.show()
#             square = np.array(square2) / 255.0
#             square[square > 0.3] = 1
#             square = np.reshape(square, newshape=(32, 32))
#             prediction = list(model.predict(np.asarray([square]))[0])
#             print(prediction)
#             print(prediction.index(max(prediction)), max(prediction))
#             # cropped_img = np.copy(image)
#             # img = cropped_img[c[1]: c[1] + 32, c[0]: c[0] + 32]
#             cv2.imshow('result', result)
#             cv2.waitKey(1000)
#             time.sleep(5)
#             square2.close()


def create_model():
    model = Sequential()

    model.add((Conv2D(60, (5, 5), input_shape=(32, 32, 1), padding='Same', activation='relu')))
    model.add((Conv2D(60, (5, 5), padding="same", activation='relu')))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.4))

    model.add((Conv2D(30, (3, 3), padding="same", activation='relu')))
    model.add((Conv2D(30, (3, 3), padding="same", activation='relu')))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    # Compiling the model
    optimizer = Adam(learning_rate=0.002)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    global train_img
    global train_classes

    # print(train_img.shape)

    history = model.fit(datagen.flow(train_img, train_classes, batch_size=32),
                        epochs=30, steps_per_epoch=200,
                        validation_data=(test_img, test_classes))

    model.save("digits_model.h5")

def crop_cell(cell):
    cell = cv2.resize(cell, (50, 50))
    cropped = cell[9:41, 9:41]
    return cropped

def create_board_matrix(image):
    cells = splitcells(image)
    model = keras.models.load_model("digits_model.h5")
    matrix = []
    for ind, cell in enumerate(cells):
        cropped_cell = crop_cell(cell).astype(float)
        cropped_cell /= 255.0
        cropped_cell = np.reshape(cropped_cell, newshape=(1, 32, 32, 1))
        # cv2.imshow(f"cell {ind}", cell)
        predictions = list(model.predict(cropped_cell)[0])
        prob = max(predictions)
        print(predictions)
        if prob > 0.4:
            matrix.append(predictions.index(prob))
        else:
            matrix.append(0)
        # cv2.waitKey()
    matrix = np.array(matrix)
    matrix = np.reshape(matrix, newshape=(9, 9))
    return matrix


def solve(table):
    find = find_empty(table)
    if not find:
        return True
    else:
        row, col = find
    for i in range(1, 10):
        if valid(table, i, (row, col)):
            table[row][col] = i
            if solve(table):
                return True
            table[row][col] = 0
    return False


def find_empty(table):
    for i in range(len(table)):
        for j in range(len(table[0])):
            if table[i][j] == 0:
                return (i, j)


def valid(table, num, pos):
    for i in range(len(table[0])):
        if table[pos[0]][i] == num and pos[1] != i:
            return False
    for i in range(len(table)):
        if table[i][pos[1]] == num and pos[0] != i:
            return False
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if table[i][j] == num and (i,j) != pos:
                return False
    return True

if __name__ == '__main__':
    # load_digits()
    # split_data()
    # preprocess_data()
    # create_matrix_classes()
    # create_model()

    image = cv2.imread('proba4.jpg')
    threshold, image = preprocess_board(image)
    image, cnt1, cnt2 = find_contours(image, threshold)
    image = do_crop(image, cnt1, cnt2)
    # cv2.imshow('prvo naslov', image)
    # cv2.waitKey()

    matrix = create_board_matrix(image)
    solve(matrix)
    print(matrix)
