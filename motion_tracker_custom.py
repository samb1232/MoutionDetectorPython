import os
import random

import cv2
import numpy as np

from custom_methods import resize_frame, convert_to_grayscale, gaussian_blur, abs_diff, threshold, dilate, \
    find_contours, bounding_rect, contour_area
from drawing_methods import draw_bounding_boxes_with_id
from ext_lib.sort import Sort
from fps_counter import FpsCounter
from utils import remove_nan


class MotionTrackerCustom:
    # PARAMETERS
    GAUSSIAN_KSIZE = (15, 15)
    GAUSSIAN_SIG_MAX = 40
    DILATE_ITERATIONS = 3

    THRESHOLD_BW = 10

    MINIMAL_BOX_CONTOUR_SIZE = 500

    SORT_MAX_AGE = 400
    SORT_MIN_HITS = 7
    SORT_IOU_THRESHOLD = 0.2

    TRACKING_POINTS_TTL = 30

    def __init__(self, video_source: str, output_video_size: tuple = (1280, 720)):
        self.tracing_points_arr = []

        self.cap = cv2.VideoCapture(video_source)

        # Initialize SORT
        self.sort = Sort(max_age=self.SORT_MAX_AGE, min_hits=self.SORT_MIN_HITS, iou_threshold=self.SORT_IOU_THRESHOLD)

        # Initialize FPS counter
        self.fps_counter = FpsCounter()
        self.prev_frame_blured = None
        self.OUTPUT_VIDEO_SIZE = output_video_size

    def detect_movement(self, frame: np.ndarray):
        gray = convert_to_grayscale(frame)

        blured = gaussian_blur(gray, self.GAUSSIAN_KSIZE, self.GAUSSIAN_SIG_MAX)

        if self.prev_frame_blured is None:
            self.prev_frame_blured = blured
            return np.empty((0, 5))

        diff = abs_diff(blured, self.prev_frame_blured)

        self.prev_frame_blured = blured

        # Apply thresholding
        thresh = threshold(diff, self.THRESHOLD_BW)

        dilated = dilate(thresh, self.DILATE_ITERATIONS)

        contours = find_contours(dilated)

        detections_list = []

        for contour in contours:
            x, y, w, h = bounding_rect(contour)
            if contour_area(contour) < self.MINIMAL_BOX_CONTOUR_SIZE:
                continue
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            detections_list.append([x1, y1, x2, y2, 1])

        # Convert list to numpy array
        if len(detections_list) == 0:
            return np.empty((0, 5))

        # Remove detections that are completely inside other detections
        for i in range(len(detections_list)):
            for j in range(len(detections_list)):
                if i == j:
                    continue
                if (detections_list[i][0] >= detections_list[j][0] and
                        detections_list[i][1] >= detections_list[j][1] and
                        detections_list[i][2] <= detections_list[j][2] and
                        detections_list[i][3] <= detections_list[j][3]):
                    detections_list[i] = [1, 1, 1, 1, 1]

        return np.array(detections_list)

    def draw_tracing_points(self, frame):
        for index, point in enumerate(self.tracing_points_arr):
            id_ = point[3]
            random.seed(int(id_))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.circle(frame, (point[0], point[1]), radius=0, color=color, thickness=6)
            self.tracing_points_arr[index][2] -= 1
            if self.tracing_points_arr[index][2] <= 0:
                self.tracing_points_arr.pop(index)
        return frame

    def process_frame(self, frame) -> np.ndarray:
        frame = resize_frame(frame, self.OUTPUT_VIDEO_SIZE)

        self.fps_counter.update()

        detections_list = self.detect_movement(frame)

        res = self.sort.update(detections_list)

        # Remove rows with nan values
        res = remove_nan(res)

        boxes_track = res[:, :-1]
        boxes_ids = res[:, -1].astype(int)

        frame = draw_bounding_boxes_with_id(frame, boxes_track, boxes_ids)

        for box in res:
            id_ = box[4]
            self.tracing_points_arr.append(
                [int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2), self.TRACKING_POINTS_TTL, id_])

        frame = self.draw_tracing_points(frame)

        return frame

    def get_next_frame(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = self.process_frame(frame)
        return frame

    def print_video(self):
        while True:
            frame = self.get_next_frame()
            if frame is None:
                return

            fps = self.fps_counter.get_fps()
            cv2.putText(frame, f'FPS: {fps}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Motion tracking", frame)
            key = cv2.waitKey(10)
            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def close(self):
        if self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    m = MotionTrackerCustom(os.path.join("videos", "vid3.mp4"), (360, 640))
    m.print_video()
