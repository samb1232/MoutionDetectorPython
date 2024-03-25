import os.path
import cv2
import numpy as np

from fps_counter import FpsCounter
from sort import Sort


class MotionTracker:
    # CONSTATNS
    OUTPUT_VIDEO_SIZE = (1280, 720)

    # PARAMETERS
    GAUSSIAN_KSIZE = (15, 15)
    GAUSSIAN_SIGMAX = 40
    DILATE_ITERATIONS = 10

    MINIMAL_BOX_CONTOUR_SIZE = 500

    def __init__(self, video_source: str):
        self.cap = cv2.VideoCapture(video_source)

        # Initialize SORT
        self.sort = Sort(max_age=400, min_hits=7, iou_threshold=0.2)

        # Initialize FPS counter
        self.fps_counter = FpsCounter()
        self.prev_frame_blured = None

    def detect_movement(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blured = cv2.GaussianBlur(gray, self.GAUSSIAN_KSIZE, self.GAUSSIAN_SIGMAX)

        if self.prev_frame_blured is None:
            self.prev_frame_blured = blured
            return np.empty((0, 5))

        diff = cv2.absdiff(blured, self.prev_frame_blured)

        self.prev_frame_blured = blured

        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

        dilated = cv2.dilate(thresh, None, iterations=self.DILATE_ITERATIONS)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detections_list = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < self.MINIMAL_BOX_CONTOUR_SIZE:
                continue
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            detections_list.append([x1, y1, x2, y2, 1])

        # Convert list to numpy array
        if len(detections_list) == 0:
            detections_list = np.empty((0, 5))
        return np.array(detections_list)

    @staticmethod
    def draw_bounding_boxes_with_id(frame: np.ndarray, bboxes: np.ndarray, ids: np.ndarray) -> np.ndarray:
        for bbox, id_ in zip(bboxes, ids):
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(frame, "ID: " + str(id_), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
        return frame

    def process(self):
        # Read first frame

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.fps_counter.update()

            frame = cv2.resize(frame, self.OUTPUT_VIDEO_SIZE)
            detections_list = self.detect_movement(frame)

            res = self.sort.update(detections_list)

            boxes_track = res[:, :-1]
            boxes_ids = res[:, -1].astype(int)

            frame = self.draw_bounding_boxes_with_id(frame, boxes_track, boxes_ids)

            fps = self.fps_counter.get_fps()
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Motion tracking", frame)
            key = cv2.waitKey(10)
            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


m = MotionTracker(os.path.join("videos", "vid1.mp4"))
m.process()
