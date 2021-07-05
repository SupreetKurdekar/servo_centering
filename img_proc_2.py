import cv2
import numpy as np

cap = cv2.VideoCapture("output.mp4")
while cap.isOpened():
    ret, frame = cap.read()


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    gray = cv2.blur(gray,(5,5))

    gray = cv2.bitwise_not(gray)

    gray = np.float32(gray)

    # use goodfeaturestotrack

    corners = cv2.goodFeaturesToTrack(gray,5,0.2,15)
    corners = np.int0(corners)
    if(len(corners)>0):
        for i in corners:
            x,y = i.ravel()
            cv2.circle(frame,(x,y),3,255,-1)

    cv2.imshow('corners', frame)
    cv2.waitKey(20)

cap.release()
cv2.destroyAllWindows()