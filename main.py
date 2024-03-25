import os.path
from time import time

import cv2
import numpy as np

from fps_counter import FpsCounter
from sort import Sort


def draw_bounding_boxes_with_id(frame, bboxes, ids):
    for bbox, id_ in zip(bboxes, ids):
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(frame, "ID: " + str(id_), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)

    return frame


video_src = os.path.join("videos", "vid1.mp4")

cap = cv2.VideoCapture(video_src)

ret, frame = cap.read()
prev_frame_blur = None

# SORT
sort = Sort(max_age=400, min_hits=7, iou_threshold=0.2)

# FPS counter init
fps_counter = FpsCounter()

while ret:
    fps_counter.update()

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
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 500:
            continue
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        detections_list.append([x1, y1, x2, y2, 1])

        # SORT Tracking
    if len(detections_list) == 0:
        detections_list = np.empty((0, 5))

    res = sort.update(np.array(detections_list))

    boxes_track = res[:, :-1]
    boxes_ids = res[:, -1].astype(int)

    frame = draw_bounding_boxes_with_id(frame, boxes_track, boxes_ids)
    fps = fps_counter.get_fps()
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Motion tracking", frame)

    prev_frame_blur = blur
    ret, frame = cap.read()

    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
