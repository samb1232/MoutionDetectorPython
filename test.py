import os.path
from time import time

import cv2
import numpy as np

from sort import Sort


def is_contour_inside(contour, larger_contour):
    x, y, w, h = cv2.boundingRect(np.array(larger_contour))  # Convert to numpy array
    x1, y1, w1, h1 = cv2.boundingRect(np.array(contour))  # Convert to numpy array

    if x1 >= x and y1 >= y and x1 + w1 <= x + w and y1 + h1 <= y + h:
        return True
    else:
        return False


def draw_bounding_boxes_with_id(frame, bboxes, ids):
    for bbox, id_ in zip(bboxes, ids):
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(frame, "ID: " + str(id_), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)

    return frame


video_src = os.path.join("videos", "vid4.mp4")

cap = cv2.VideoCapture(video_src)

ret, frame = cap.read()
prev_frame_blur = None

sort = Sort(max_age=400, min_hits=7, iou_threshold=0.2)

fps_counter = 0
start_time = time()
fps = 0

while ret:
    fps_counter += 1
    cur_time = time()
    time_diff = cur_time - start_time
    if time_diff > 1.0:
        fps = fps_counter / np.round(time_diff)
        start_time = time()
        fps_counter = 0

    frame = cv2.resize(frame, [1280, 720])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 40)

    if prev_frame_blur is None:
        prev_frame_blur = blur
        continue

    diff = cv2.absdiff(blur, prev_frame_blur)

    _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, None, iterations=10)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections_list = []

    for contour in contours:
        if len(detections_list) > 0:
            for larger_contour in detections_list:
                if is_contour_inside(contour, larger_contour):
                    break
            else:
                x, y, w, h = cv2.boundingRect(contour)
                if cv2.contourArea(contour) < 3000:
                    continue
                detections_list.append([x, y, x + w, y + h, 1])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 3000:
                continue
            detections_list.append([x, y, x + w, y + h, 1])

    if len(detections_list) == 0:
        detections_list = np.empty((0, 5))

    res = sort.update(np.array(detections_list))

    boxes_track = res[:, :-1]
    boxes_ids = res[:, -1].astype(int)

    frame = draw_bounding_boxes_with_id(frame, boxes_track, boxes_ids)

    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Motion tracking", frame)

    prev_frame_blur = blur
    ret, frame = cap.read()

    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
