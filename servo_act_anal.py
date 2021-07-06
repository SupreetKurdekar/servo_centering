import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
cap = cv2.VideoCapture("output.mp4")
template = cv2.imread("template.png")
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
template=cv2.GaussianBlur(template,ksize=(5,5),sigmaX=1)
frame = cv2.GaussianBlur(frame,ksize=(5,5),sigmaX=1)
orb = cv2.ORB_create(nfeatures=10000,patchSize=40)
kp1 = orb.detect(template,None)
kp1, des1 = orb.compute(template, kp1)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(template, kp1, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()
kp2 = orb.detect(frame,None)
kp2, des2 = orb.compute(frame, kp2)
# draw only keypoints location,not size and orientation
img3 = cv2.drawKeypoints(frame, kp2, None, color=(0,255,0), flags=0)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
img4 = np.zeros((480,640,3))

src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches[:500]]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches[:500] ]).reshape(-1,1,2)
# Draw first 10 matches.
img4=cv2.drawMatches(template,kp1,frame,kp2,matches[:500],img4,flags=2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h, w, d = template.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
img5 = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

plt.imshow(img5),plt.show()
plt.imshow(img3), plt.show()
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
