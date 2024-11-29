import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    cv2.imshow("Video", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

