import random

import cv2
import numpy as np


def draw_bounding_boxes_with_id(frame: np.ndarray, bboxes: np.ndarray, ids: np.ndarray) -> np.ndarray:
    for bbox, id_ in zip(bboxes, ids):
        random.seed(int(id_))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        # cv2.putText(frame, "ID: " + str(id_), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.9, color, 2)
    return frame

