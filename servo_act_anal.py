import numpy as np
import cv2
import utils

cap = cv2.VideoCapture("output.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    break
    # if frame is read correctly ret is True
    # if not ret:
    #     print("Can't receive frame (stream end?). Exiting ...")
    #     break
    # # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break
cv2.imshow("fr",frame)
result = utils.black_white(frame)
# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
# # Threshold of blue in HSV space
# lower_blue = np.array([0, 0, 200])
# upper_blue = np.array([180, 255, 30])
#
# # preparing the mask to overlay
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
#
# # The black region in the mask has the value of 0,
# # so when multiplied with original image removes all non-blue regions
# result = cv2.bitwise_and(frame, frame, mask=mask)
cv2.imshow("fr2",result)
cv2.waitKey(0)
