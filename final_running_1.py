import cv2
import numpy as np
from scipy.spatial.distance import cdist

def get_line_from_image(image):

    points = [[1,0],[3,0],[7,0],[25,0]]
    points = np.array(points)


    d = cdist(points,points)
    # print(d)

    sorted = np.sort(d.ravel())
    # print(sorted)

    top = sorted[5:9]
    # print(top)

    inds1 = np.where(d==top[0])
    inds2 = np.where(d==top[3])

    # print(inds1,inds2)

    vec1 = np.mean(points[inds1[0],:],axis=0)
    vec2 = np.mean(points[inds2[0],:],axis=0)

    # print(vec1,points[inds1[0],:],vec2,points[inds2[0],:])

    return vec1-vec2

def get_angle(vec1,vec2):
    vec1 = vec1/np.linalg.norm(vec1)
    vec2 = vec2/np.linalg.norm(vec2)

    dot = np.dot(vec1,vec2)

    theta = np.argcos(dot/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))

    return theta

