import cv2
import numpy as np

prev_img = np.zeros((480, 640), dtype = "uint8")
# pre_prev_img = np.zeros((480, 640), dtype = "uint8")

prev_diff = np.zeros((480, 640), dtype = "uint8")

cap = cv2.VideoCapture("output_original.mp4")
while cap.isOpened():           
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray before smoothing",gray)

    # this filter removes shadowy/dark regions
    # since we know that the important object is very bright
    ret,intensity_filter = cv2.threshold(gray,80,1,cv2.THRESH_BINARY_INV)
    cv2.imshow("intensity_mask",intensity_filter)

    gray = gray*intensity_filter

    cv2.imshow("gray after intensity",gray)

    #gray = cv2.blur(gray,(25,25))
    # gray = cv2.GaussianBlur(gray,(7,7),0)

    # adapt_thr = cv2.adaptiveThreshold(gray,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    # gray = gray*adapt_thr

    cv2.imshow("gray after smothing",gray)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_range = np.array([105,50,50])
    upper_range = np.array([135,255,255])

    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = (mask/255).astype(np.uint8)

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5,5), np.uint8)

    dilated_blue_mask = cv2.dilate(mask, kernel, iterations=17)
    cv2.imshow("blue_mask",dilated_blue_mask)

    gray = gray*dilated_blue_mask

    cv2.imshow("after roi",gray)

    corners = cv2.goodFeaturesToTrack(gray,100,0.5,15)
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

    # ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, ker)
    # dub_diff = cv2.morphologyEx(dub_diff, cv2.MORPH_OPEN, ker)

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