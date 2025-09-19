import numpy as np
import cv2

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel = cv2.Sobel(blur, cv2.CV_64F, 1, 1, ksize=1)

    cv2.imwrite('sobel.png', sobel)

def canny_edge_detection(image, image_threshold1=50, image_threshold2=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blur, image_threshold1, image_threshold2)

    cv2.imwrite('canny.png', canny)

def template_match(image, template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img', img_gray)
    #cv2.waitKey(0)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('template', template_gray)
    #cv2.waitKey(0)
    w, h = template_gray.shape[::-1]
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('template_matched.png', image)


def resize(image, scale_factor: int, up_or_down: str):
    if up_or_down == 'up':
        for _ in range(scale_factor):
            image = cv2.pyrUp(image)
    elif up_or_down == 'down':
        for _ in range(scale_factor):
            image = cv2.pyrDown(image)

    name = 'resized_' + up_or_down + str(scale_factor) + '.png'

    cv2.imwrite(name, image)


if __name__ == "__main__":
    image = cv2.imread("lambo.png")
    shapes = cv2.imread("shapes.png")
    template = cv2.imread("shapes_template.jpg")

    sobel_edge_detection(image)
    canny_edge_detection(image, 50, 50)
    template_match(shapes, template)

    resize(image, 1, "up")
    resize(image, 1, "down")
    resize(image, 2, "up")
    resize(image, 2, "down")