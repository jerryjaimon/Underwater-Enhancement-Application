import cv2
import numpy as np
from matplotlib import pyplot as plt

def contrast_stretching(img,a,b,c,d):
    img1 = img
    '''for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img1[i][j] = (((img[i][j] - c) * (b - c))/ (d - c)) + a'''
    val1 = b-c
    val2 = d-c
    img1 = cv2.add((img-c)*val1/val2,a)
    return img1

def displayoutput(title, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Title', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def underwater_integrated_color_model(path,location):
    img=cv2.imread(path)
    blue_channel,green_channel,red_channel  = cv2.split(img)

    blue_min = np.min(blue_channel)
    red_min = np.min(red_channel)
    green_min = np.min(green_channel)

    blue_max = np.max(blue_channel)
    red_max = np.max(red_channel)
    green_max = np.max(green_channel)

    desired_max = 200
    desired_min = 0

    red_channel = cv2.equalizeHist(red_channel)
    green_channel = cv2.equalizeHist(green_channel)
    blue_channel = cv2.equalizeHist(blue_channel)
    contrast_image = np.zeros(img.shape, dtype='uint8')
    contrast_image[:, :, 0] = blue_channel
    contrast_image[:, :, 1] = green_channel
    contrast_image[:, :, 2] = red_channel
    hls_image = cv2.cvtColor(contrast_image, cv2.COLOR_BGR2HLS)

    hls_image[:, :, 1] = cv2.equalizeHist(hls_image[:, :, 1])
    hls_image[:, :, 2] = cv2.equalizeHist(hls_image[:, :, 2])

    bgr = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR_FULL)
    cv2.imwrite(location,bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()