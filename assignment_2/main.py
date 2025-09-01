import cv2 as cv
import numpy as np


def padding(image, border_width):
    return cv.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv.BORDER_REFLECT)

def crop(image, x_0, x_1, y_0, y_1):
    return image[y_0:y_1, x_0:x_1]

def resize(image, width, height):
    return cv.resize(image, (width, height))

def copy(image, emptyPictureArray):
    height, width, _ = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            emptyPictureArray[i, j] = image[i, j]
    return emptyPictureArray

def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def hsv(image):
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)

def hue_shifted(image, emptyPictureArray, hue):
    image = hsv(image)
    height, width, _ = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            h, s, v = image[i, j]
            h = (h + hue) % 180
            emptyPictureArray[i, j] = [h, s, v]

    return cv.cvtColor(emptyPictureArray, cv.COLOR_HSV2BGR)

def smoothing(image):
    ksize = (15, 15)
    return cv.GaussianBlur(image, ksize, 0)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv.rotate(image, cv.ROTATE_180)

if __name__ == "__main__":

    image = cv.imread("lena.png")

    padding_image = padding(image, 100)
    cv.imwrite("padding.png", padding_image)

    cropped_image = crop(image, 80, image.shape[1]-130, 80, image.shape[0]-130)
    cv.imwrite("cropped.png", cropped_image)

    resized_image = resize(image, 200, 200)
    cv.imwrite("resized.png", resized_image)

    emptyPictureArray = None
    copied_image = copy(image, emptyPictureArray)
    cv.imwrite("copied.png", copied_image)

    grayscale_image = grayscale(image)
    cv.imwrite("grayscale.png", grayscale_image)

    hsv_image = hsv(image)
    cv.imwrite("hsv.png", hsv_image)

    emptyPictureArray = None
    hue_shifted_image = hue_shifted(image, emptyPictureArray, 50)
    cv.imwrite("hue_shifted.png", hue_shifted_image)

    smoothed_image = smoothing(image)
    cv.imwrite("smoothed.png", smoothed_image)

    rotated_image_90 = rotation(image, 90)
    cv.imwrite("rotated_90.png", rotated_image_90)
    rotated_image_180 = rotation(image, 180)
    cv.imwrite("rotated_180.png", rotated_image_180)