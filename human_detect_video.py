import torch
import cv2
import time
from my_lib import *

if __name__ == "__main__":

    # Code execution start time
    start = time.time()

    # Creating a dictionary to save each target object's ID ({id: center of object's bbox})
    id_boxes = {}

    # Start ID count
    id_count = 1

    # Build detect model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Detect only humans (YOLOv5 standart model)
    model.classes = [0]

    # Setting target object name for text variables
    target_object = 'Humans at this moment'

    # Import video
    video = cv2.VideoCapture('people_walking.mp4')

    # Getting frame video size
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Setting video format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Build video output
    output_video = cv2.VideoWriter('human_output.mp4', fourcc, 30, (width, height), isColor=True)

    # Reset frame number for the function
    frame_number = 1

    print('Wait until the end of the inference...')

    # Reading video
    while video.isOpened():
        success, frame = video.read()
        if success == False:
            break

        # Detect and tracking humans in the frames
        frame, id_count = detect_and_track(frame, model, frame_number, id_boxes, id_count, target_object)

        # Monitoring the process
        print(f'Frame {frame_number}: ok')

        # Writing frames in the video output
        output_video.write(frame)

        # Incrementing the frame number
        frame_number += 1

    # Releasing the input and output video
    video.release()
    output_video.release()

    print('Inference completed successfully!')

    # Code execution end time
    end = time.time()
    print(f'Time taken to run the code: %.3f seconds' % (end-start))
