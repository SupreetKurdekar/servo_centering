import cv2
import numpy as np

cap = cv2.VideoCapture("output.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    break

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_range = np.array([105,50,50])
upper_range = np.array([135,255,255])

mask = cv2.inRange(hsv, lower_range, upper_range)


# Taking a matrix of size 5 as the kernel
kernel = np.ones((3,3), np.uint8)

img_dilation = cv2.dilate(mask, kernel, iterations=10)

cv2.imshow("dilated_blue_mask",img_dilation)

gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)



gray = cv2.blur(gray,(5,5))


gray = cv2.bitwise_not(gray)


gray = np.float32(gray)


# harris corner detection did not really work
# too many noisy corners
# dst = cv2.cornerHarris(gray,5,5,0.2)

# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)

# # Threshold for an optimal value, it may vary depending on the image.
# frame[dst>0.01*dst.max()]=[0,0,255]

# use goodfeaturestotrack

corners = cv2.goodFeaturesToTrack(gray,5,0.2,15)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(frame,(x,y),3,255,-1)


hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of white color in HSV
# change it according to your need !
lower_white = np.array([0,0,0], dtype=np.uint8)
upper_white = np.array([0,0,255], dtype=np.uint8)

# Threshold the HSV image to get only white colors
mask = cv2.inRange(hsv, lower_white, upper_white)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask= mask)

cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)


while(True):
   k = cv2.waitKey(5) & 0xFF
   if k == 27:
      break

cv2.destroyAllWindows()