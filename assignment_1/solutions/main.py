import cv2
import numpy as np

def print_image_information(image):

    height, width, channels = image.shape

    print("Image Information:")
    print(f"Height = {height}")
    print(f"Width = {width}")
    print(f"Channels = {channels}")
    print(f"Size = {image.size}")
    print(f"Datatype = {image.dtype}")

if __name__ == "__main__":

    # part IV assignment 1
    img = cv2.imread("image.png")

    print_image_information(img)


    # part V assignment 1
    cam= cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cam.get(cv2.CAP_PROP_FPS)

    with open('camera_outputs.txt', 'w') as f:
        f.write(f"Width: {frame_width}\n")
        f.write(f"Height: {frame_height}\n")
        f.write(f"FPS: {fps}\n")