import os
import keras.models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sudoku import Sudoku
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import math
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


def prep(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


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


def preprocess_board(image):
    image = cv2.resize(image, (450, 450))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_image, (3, 3), 6)
    threshold_img = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return threshold_img, image


def find_board(contour):
    maxi = np.array([])
    max_area = 0
    for i in contour:
        # nadji povrsinu konture
        area = cv2.contourArea(i)
        if area > 50:
            # proveri duzinu konture
            peri = cv2.arcLength(i, True)
            # aproksimiraj je poligonom
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                maxi = approx
                max_area = area
    return maxi


def split_cells(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for box in cols:
            boxes.append(box)
    return boxes


def sort_points(points):
    points = points.reshape((4, 2))
    sorted_points = np.zeros((4, 1, 2), dtype=np.int32)

    mini = math.inf
    mini_ind = -1
    maxi = -math.inf
    maxi_ind = -1

    for ind, point in enumerate(points):
        suma = point[0] + point[1]
        if suma < mini:
            mini_ind = ind
            mini = suma
        if suma > maxi:
            maxi_ind = ind
            maxi = suma

    sorted_points[0] = points[mini_ind]
    sorted_points[3] = points[maxi_ind]

    mini = math.inf
    mini_ind = -1
    maxi = -math.inf
    maxi_ind = -1

    for ind, point in enumerate(points):
        diff = point[0] - point[1]
        if diff < mini:
            mini_ind = ind
            mini = diff
        if diff > maxi:
            maxi_ind = ind
            maxi = diff

    sorted_points[1] = points[maxi_ind]
    sorted_points[2] = points[mini_ind]

    return sorted_points


def crop_board(image, cnt1, cnt2):
    biggest = find_board(cnt1)
    if biggest.size != 0:
        biggest = sort_points(biggest)
        cv2.drawContours(cnt2, biggest, -1, (0, 255, 0), 10)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imagewrap = cv2.warpPerspective(image, matrix, (450, 450))
        imagewrap = cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)

        return imagewrap
    return None


def find_contours(image, threshold):
    image1 = image.copy()
    image2 = image.copy()
    cnt, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image1, cnt, -1, (0, 255, 0), 3)
    return image, cnt, image2


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


def create_model():
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
                        epochs=30, steps_per_epoch=200,
                        validation_data=(test_img, test_classes))

    model.save("digits_model.h5")


def crop_cell(cell):
    cell = cv2.resize(cell, (50, 50))
    cropped = cell[9:41, 9:41]
    return cropped


def create_board_matrix(image):
    cells = split_cells(image)
    model = keras.models.load_model("digits_model.h5")
    matrix = []
    for ind, cell in enumerate(cells):
        cropped_cell = crop_cell(cell).astype(float)
        cropped_cell /= 255.0
        cropped_cell = np.reshape(cropped_cell, newshape=(32, 32, 1))
        blank_cell = np.count_nonzero(cropped_cell < 0.4) < 20

        cropped_cell = np.array([cropped_cell])

        predictions = list(model.predict(cropped_cell)[0])
        prob = max(predictions)
        print(predictions)

        if blank_cell:
            print(f"BLANK AT {ind}")
            matrix.append(0)
            continue

        if prob >= 0.3:
            matrix.append(predictions.index(prob) + 1)
        else:
            matrix.append(0)

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

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if table[i][j] == num and (i, j) != pos:
                return False
    return True


def check_solution(matrix):
    board = []

    for array in matrix:
        list = []
        for num in array:
            list.append(int(num))
        board.append(list)

    puzzle = Sudoku(3, 3, board)

    solve(matrix)

    try:
        puzzle = puzzle.solve(raising=True)
    except Exception:
        print("Couldn't solve puzzle")
        return

    equal = True

    for i in range(9):
        for j in range(9):
            if int(puzzle.board[i][j]) != int(matrix[i][j]):
                print(f'Difference {puzzle.board[i][j]} {int(matrix[i][j])}')
                equal = False
    if equal:
        print("Solution same as the one given with pysudoku")

    else:
        print("Solution not the same as the one given with pysudoku")
    print(matrix)


if __name__ == '__main__':
    # load_digits()
    # preprocess_data()
    # create_matrix_classes()
    # create_model()

    image = cv2.imread("primer12.jpg")
    threshold, image = preprocess_board(image)
    image, cnt1, cnt2 = find_contours(image, threshold)
    image = crop_board(image, cnt1, cnt2)

    matrix = create_board_matrix(image)
    print(matrix)
    check_solution(matrix)
