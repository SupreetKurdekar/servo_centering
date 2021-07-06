import cv2
import numpy as np

prev_img = np.zeros((480, 640), dtype = "uint8")
# pre_prev_img = np.zeros((480, 640), dtype = "uint8")

prev_diff = np.zeros((480, 640), dtype = "uint8")

cap = cv2.VideoCapture("output_original.mp4")
while cap.isOpened():           
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #gray = cv2.blur(gray,(25,25))
    # gray = cv2.GaussianBlur(gray,(6,6),0)

    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    cv2.imshow("gray after smoothing",gray)

    corners = cv2.goodFeaturesToTrack(gray,25,0.2,15)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(frame,(x,y),3,255,-1)

    # # Otsu's thresholding after Gaussian filtering
    # gray = cv2.GaussianBlur(gray,(7,7),0)
    # ret3,gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # gray = cv2.bitwise_not(gray)

    gray = np.float32(gray)

    diff = gray - prev_img

    dub_diff = diff - prev_diff

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, ker)
    dub_diff = cv2.morphologyEx(dub_diff, cv2.MORPH_OPEN, ker)

    # smooth_diff = cv2.blur(gray,(5,5))

    prev_img = gray
    
    prev_diff = diff

    # use goodfeaturestotrack



    cv2.imshow('diff', diff)
    cv2.imshow('dub_diff', dub_diff)
    cv2.imshow("current",frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()