import torch
import cv2
import math

# PARAMETERS

# frame - frame that the function will be applied
# model - detect model that will be applied in each frame
# frame_number - number of current frame
# id_boxes - dictionary where the IDs will be saved
# id_count - counter for new IDs detected
# target_object - string for text variables


def detect_and_track(frame, model, frame_number, id_boxes, id_count, target_object):

    # Takin the frame shape
    h, w, c = frame.shape

    # Defining some text settings
    text_size = round(min(w, h)/666, 1)
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1

    # Defining minimum distance for Euclidean distance calculation
    min_dist = round(min(w, h)/27)

    # Applying the detect model
    results = model(frame)

    # Getting the model output information
    results_data = results.pandas().xyxy[0]

    # Getting the amount of target object in the frame
    target_object_amount = results_data.shape[0]

    # For each bbox in the frame
    for i in range(target_object_amount):

        # Taking coordinates of bbox
        x1, y1 = round(results_data['xmin'][i]), round(results_data['ymin'][i])
        x2, y2 = round(results_data['xmax'][i]), round(results_data['ymax'][i])

        # Drawing the bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Getting bbox centers
        cx, cy = round(x1 + (x2-x1)/2), round(y1 + (y2-y1)/2)

        # For the first frame
        if frame_number == 1:

            # Save the ID and bbox center of the first frame in the dictionary
            id_boxes.update({str(id_count): (cx, cy)})

            # Writing ID with background in bbox
            id_with_bg(frame, w, h, id_count, (x1, y1), text_font, 0.6 *
                       text_size, (255, 255, 255), text_thickness, (255, 0, 0))

            # Incrementing the ID count to the next new object
            id_count += 1

        # For the following frames
        if frame_number > 1:

            # Setting the new object detection variable
            new_object = False

            # Comparing the bbox of the new frame with the previous one
            for id, center in id_boxes.items():

                # Distance between the two centers, new and previous (Euclidean distance)
                dist = math.hypot(cx - center[0], cy - center[1])

                # If the Euclidean distance is less than the minimum distance,
                # then the current box probably refers to the same object as in the previous frame.
                if dist < min_dist:

                    # Updating the new object detection variable
                    new_object = True

                    # Updating ID and bbox that are in the dictionary
                    id_boxes.update({str(id): (cx, cy)})

                    # Writing ID with background in bbox
                    id_with_bg(frame, w, h, id, (x1, y1), text_font,
                               0.6*text_size, (255, 255, 255), text_thickness, (255, 0, 0))

                    break

            # If bbox refers to a new detected object
            if new_object == False:

                # Add the new ID with the center of the new object to the dictionary
                id_boxes.update({str(id_count): (cx, cy)})

                # Writing ID with background in bbox
                id_with_bg(frame, w, h, id_count, (x1, y1), text_font,
                           0.6*text_size, (255, 255, 255), text_thickness, (255, 0, 0))

                # Incrementing the ID count to the next new object
                id_count += 1

    # Getting size of object label text in scale
    ((txt_w, txt_h), _) = cv2.getTextSize(
        f'{target_object}: {str(target_object_amount).zfill(2)}', text_font, text_size, text_thickness)

    # Start point of object label
    x = round(w-(txt_w + 0.02*w))
    y = round(txt_h+0.03*h)

    # Object label background settings
    txt_bgstart = (x - round(0.02*txt_w), y-round(1.3*txt_h))
    txt_bgend = (x + round(1.02*txt_w), y + round(0.3*txt_h))

    # Apllying the object label background
    cv2.rectangle(frame, txt_bgstart, txt_bgend, (0, 0, 0), -1)

    # Putting object label in the top right of screen
    cv2.putText(frame, f'{target_object}: {str(target_object_amount).zfill(2)}',
                (x, y), text_font, text_size, (0, 255, 0), text_thickness)

    return frame, id_count


def id_with_bg(frame, w, h, id, points, id_font, id_size, id_color, id_thickness, bg_color):

    # (Function to facilitate the placement of IDs with background)

    # Putting ID object in the text label
    text = f'ID: {str(id).zfill(2)}'

    # Getting size of ID object text in scale
    ((id_w, id_h), _) = cv2.getTextSize(text, id_font, id_size, id_thickness)

    # Start point of ID object
    x = round(w-(id_w + 0.02*w))
    y = round(id_h+0.03*h)

    # ID object background settings
    bg_start = (points[0] - round(0.02*id_w), points[1]-round(1.3*id_h))
    bg_end = (points[0] + round(1.02*id_w), points[1] + round(0.3*id_h))

    # Apllying the ID object background
    cv2.rectangle(frame, bg_start, bg_end, bg_color, -1)

    # Putting ID object in the top left of bboxes
    cv2.putText(frame, text, (points[0], points[1]), id_font, id_size, id_color, id_thickness)
