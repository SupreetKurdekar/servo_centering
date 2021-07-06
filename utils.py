import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def coords_in_img(pts,M):
    pts = np.array([[0,0],[0,pts.shape[1]],[pts.shape[0],pts.shape[1]],[pts.shape[0],0]])
    pts = np.hstack((pts,np.ones((pts.shape[0],1))))
    pts_new = np.matmul(M,pts.transpose())
    pts_new = pts_new.transpose()[:,:2]
    return pts_new
def get_line_from_image(points):
    theta = np.pi/2
    # R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    # points = np.matmul(R,points.transpose()).transpose()
    # points = [[1,0],[3,0],[7,0],[25,0]]
    # points = np.array(points)
    d = cdist(points,points)
    # print(d)
    sorted = np.sort(d.ravel())
    # print(sorted)
    top = sorted[5:9]
    # print(top)
    inds1 = np.where(d==top[0])
    inds2 = np.where(d==top[3])
    # print(inds1,inds2)
    pts = np.vstack((np.mean(points[[inds1[0][0],inds1 [1][0]],:],axis=0),np.mean(points[[inds2[0][0],inds2[1][0]],:],axis=0)))
    pts = pts[np.argsort(pts[:,0]),:]
    return pts.reshape(2,2)
    # print(vec1,points[inds1[0],:],vec2,points[inds2[0],:])
    # if pts[0,1]>pts[1,1]:
    #     vec = pts[0,:]-pts[1,:]
    # else:
    #     vec = -(pts[0,:]-pts[1,:])
    # return pts[0,:]-pts[1,:]

def get_angle(pts1,pts2):
    vec1 = pts1[0,:]-pts1[1,:]
    if pts1[0,1]>pts2[0,1]:
        vec2 = pts2[0,:]-pts2[1,:]
    else:
        vec2 = -(pts2[0,:]-pts2[1,:])
    vec1 = vec1/np.linalg.norm(vec1)
    vec2 = vec2/np.linalg.norm(vec2)
    dot = np.dot(vec1,vec2)
    theta = np.arccos(dot/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return theta*57.2958

def get_line_pts(dst):
    dst_line = np.polyfit(dst[:,0],dst[:,1],deg=1)
    fit = lambda c,x: c[0]*x+c[1]
    # src_pts = np.vstack((src_line(np.min(src[:,0])),src_line(np.max(src[:,0]))))
    dst_pts = np.array([[np.min(dst[:, 0]),fit(dst_line,np.min(dst[:, 0]))], [np.max(dst[:, 0]),fit(dst_line,np.max(dst[:, 0]))]])
    return dst_pts

# jk = np.array([[-1,0],[-1,1],[2,0],[2.5,1]])
# print(jk.shape)
# c= np.mean(jk,axis=0)
# theta = np.pi/4
# R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
# jk2 = jk-c
# jk2 = np.matmul(R,jk2.transpose())
# jk2 = jk2.transpose()+c
# # jk2 = np.empty((4,2))
# # jk2[:,0] = ((jk[:,0] - c[0]) * np.cos(theta) + (jk[:,1]  - c[1]) * np.sin(theta)) + c[0]
# # jk2[:,1]= (-(jk[:,0] - c[0]) * np.sin(theta) + (jk[:,1]  - c[1]) * np.cos(theta)) +  c[0]
# pts1 = get_line_from_image(jk)
# pts2 = get_line_from_image(jk2)
# plt.figure()
# plt.scatter(x=jk[:,0],y=jk[:,1])
# plt.scatter(x=jk2[:,0],y=jk2[:,1])
# plt.show()
# ang = get_angle(pts1,pts2)
# print(ang)

"""Black and white processing"""


def black_white(img):
    # for image_path in images_path:
    #     img = cv2.imread(image_path)
        # img = cv2.resize(img, (64, 32), interpolation=cv2.INTER_CUBIC)
    height, width, depth = np.array(img).shape
    white = [255, 255, 255]
    black = [0, 0, 0]
    # Turn non-white areas to white
    # for i in range(height):
    #     for j in range(width):
    #         if white in img[i, j, :]:
    #             pass
    #         else:
    #             img[i, j, :] = black
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.waitKey(0)
    ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cnts,_ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(cnts))
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # Sort the outlines in descending order by area
    c = sorted(cnts, key=cv2.contourArea, reverse=True)
    largest = c[0]

    img2 = np.zeros((480,640)).astype(np.uint8)  # create a single channel 200x200 pixel black image
    cv2.fillPoly(img2, pts=[largest], color=(255, 255, 255))
    res = cv2.bitwise_and(img, img, mask=img2)
    cv2.imshow("gray",res)

    # Fill small areas with black
    # for i in range(len(c),len(c)+1):
    # x, y, w, h = cv2.boundingRect(largest)
    # img[y:(y + h), x:(x + w), :] = black
    return  res
    # cv2.rectangle(imgsize, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imwrite('./output_disk_data' + '/' + str(image_path.split('/')[2]) + '.jpg', img)
        # output_path(output_path, image_path, img)


# if __name__ == '__main__':
#     black_white(images_path, output_path)