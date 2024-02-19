import os.path

import cv2, time

video_src = os.path.join("videos", "video.mp4")

cap = cv2.VideoCapture(video_src)

ret, frame = cap.read()
prev_frame = None

while ret:
    frame = cv2.resize(frame, [1280, 720])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    if prev_frame is None:
        prev_frame = blur
        continue

    delta_frame = cv2.absdiff(blur, prev_frame)

    cv2.imshow("Video", delta_frame)

    prev_frame = blur
    ret, frame = cap.read()

    key = cv2.waitKey(20)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
