import cv2
import time
def displayoutput(title, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Title', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clahe_fun(path,location):
    image=cv2.imread(path)
    bgr=image
    gridsize=3
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(location, clahe_img)