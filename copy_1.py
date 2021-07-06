import numpy as np
import cv2 as cv
import  utils
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 5
img1 = cv.imread('template.png')
cap = cv.VideoCapture("output_original.mp4")
count =0
img1 = cv.GaussianBlur(img1,ksize=[5,5],sigmaX=1)
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
# queryImage
    img2 = frame # trainImage
    img2 = cv.GaussianBlur(img2, ksize=[5, 5],sigmaX=1)
# Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)
    # flann = cv.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1,des2,k=2)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    good = matches[:30]
    # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)
    # Now we set a condition that atleast 10 matches (defined by MIN_MATCH_COUNT) are to be there to find the object. Otherwise simply show a message saying not enough matches are present.

    # If enough matches are found, we extract the locations of matched keypoints in both the images. They are passed to find the perspective transformation. Once we get this 3x3 transformation matrix, we use it to transform the corners of queryImage to corresponding points in trainImage. Then we draw it.
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    ## Act Computation
    # pts = utils.coords_in_img(img1,M)
    pts = dst.reshape(dst.shape[0],2)
    if count==0:
        angle=[]
        pts_ref = [utils.get_line_from_image(pts)]#[np.array([[0,0],[1,0]])]#
    else:
        # pts_cur = utils.get_line_from_image(pts)
        pts_cur = utils.get_line_pts(pts)
        angle.append(utils.get_angle(pts_ref[-1].copy(),pts_cur))
        print(angle[-1])
        if angle[-1]<=3 or angle[-1]>=177:
            pts_ref[-1]=(0.5*(pts_ref[-1]+pts_cur))#pts_ref[-1]#
        else:
            print("motion_detected")
            pts_ref.append(pts_cur)

    # Finally we draw our inliers (if successfully found the object) or matching keypoints (if failed).
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    coords = [img1.shape[0],img1.shape[1]]
    cv.imshow("frame",img3)
    cv.waitKey(0)
    count+=1
cap.release()
plt.scatter(x=range(len(angle)),y=angle)
plt.show()