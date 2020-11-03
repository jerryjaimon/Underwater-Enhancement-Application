import cv2
import numpy as np
import time

def contrast_stretching(img,a,b,c,d):
    img1 = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img1[i][j] = (((img[i][j] - c) * (b - c))/ (d - c)) + a
    return img1

def displayoutput(title, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Title', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def AUV150(path,location):
    start = time.time()
    image = cv2.imread(path)
    # Box Filter
    kernal_size = (3, 3)
    box_blur = cv2.blur(image, kernal_size)
    gaussian_blur = cv2.GaussianBlur(image, kernal_size, 0)
    kernal_s = 3
    median_blur = cv2.medianBlur(image, kernal_s, 0)
    median_after = cv2.medianBlur(box_blur, 3, 0)
    ####################################################
    ###### Contrast Stretching in RGB color space ######
    ####################################################
    img = median_after
    blue_channel,green_channel,red_channel = cv2.split(img)
    blue_min = np.min(blue_channel)
    red_min = np.min(red_channel)
    green_min = np.min(green_channel)
    blue_max = np.max(blue_channel)
    red_max = np.max(red_channel)
    green_max = np.max(green_channel)

    desired_max = 255
    desired_min = 0

    red_channel = cv2.equalizeHist(red_channel)
    green_channel = cv2.equalizeHist(green_channel)
    blue_channel = cv2.equalizeHist(blue_channel)

    contrast_image = np.zeros(img.shape, dtype='uint8')
    contrast_image[:, :, 0] = blue_channel
    contrast_image[:, :, 1] = green_channel
    contrast_image[:, :, 2] = red_channel
    ####Clahe
    bgr = contrast_image
    gridsize = 2
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(gridsize, gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    ##Color preserving processing in YCbCr color space##
    brightYCB = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2YCrCb)
    y = brightYCB[:, :, 0]
    y = cv2.equalizeHist(y)
    brightYCB[:, :, 0] = y
    final_img = cv2.cvtColor(brightYCB, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(location, final_img)
    cv2.imshow("Paper-1", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

