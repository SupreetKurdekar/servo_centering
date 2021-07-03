import imutils
import scipy.spatial
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# images_path = './new_pig_data'
# # images_path='./new_disk_data'
#
# output_path = './output_pig_data'
# # output_path='./output_disk_data'
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
# """Read Picture Path"""
#
#
# def read_path(image_path):
#     images_name = os.listdir(image_path)
#     images_path = [os.path.join(image_path, image_name) for image_name in images_name]
#     print(images_path)
#     print('./output_data' + images_path[0].split('/')[2].replace(' ', '') + '.jpg')
#     return images_path
#
#
# """Output Picture"""
#
#
# def output_path(output_path, image_path, img):
#     cv2.imwrite(output_path + '/' + image_path.split('/')[2].replace(' ', ''), img)


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