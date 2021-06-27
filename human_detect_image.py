import torch
import cv2
import time
from my_lib import *

if __name__ == "__main__":

    # Creating a dictionary to save each target object's ID ({id: center of object's bbox})
    id_boxes = {}

    # Start ID count
    id_count = 1

    # Code execution start time
    start = time.time()

    # Build model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Detect only humans (YOLOv5 standart model)
    model.classes = [0]

    # Setting target object name for text variables
    target_object = 'Humans at this moment'

    # Import image
    img = cv2.imread('football.jpg')

    # Setting frame number for the function (Image = 1 frame)
    frame_number = 1

    # Detect humans on the image
    frame, id_count = detect_and_track(img, model, frame_number, id_boxes, id_count, target_object)

    # Show image on screen
    cv2.imshow('image', img)
    cv2.waitKey(0)

    # Code execution end time
    end = time.time()
    print(f'Time taken to run the code: %.3f seconds' % (end-start))
