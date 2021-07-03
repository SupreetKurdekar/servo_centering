import cv2
import numpy as np

cap = cv2.VideoCapture("http://192.168.1.239:8080/video")
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(frame) # frame is a numpy.ndarray with shape (1280, 720, 3)

out.release()
cap.release()
cv2.destroyAllWindows()