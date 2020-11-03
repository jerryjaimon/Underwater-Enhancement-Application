import cv2
import numpy as np
import math
import time

def Saliency(img):
    gfgbr = cv2.GaussianBlur(img,(3, 3), 3)
    LabIm = cv2.cvtColor(gfgbr, cv2.COLOR_BGR2Lab)
    lab = cv2.split(LabIm)
    l = np.float32(lab[0])
    a = np.float32(lab[1])
    b = np.float32(lab[2])
    lm = cv2.mean(l)[0] # cv2.mean(l).val[0]
    am = cv2.mean(a)[0]
    bm = cv2.mean(b)[0]
    sm = np.zeros(l.shape, l[0][1].dtype)
    l = cv2.subtract(l, lm)
    a = cv2.subtract(a, am)
    b = cv2.subtract(b, bm)
    sm = cv2.add(sm, cv2.multiply(l, l)) 
    sm = cv2.add(sm, cv2.multiply(a, a))
    sm = cv2.add(sm, cv2.multiply(b, b))
    return sm


def LaplacianContrast(img):
    # img=cv2.CreateMat(h, w, cv2.CV_32FC3)
    laplacian = cv2.Laplacian(img,5) 
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian


def LocalContrast(img):
    h = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]
    mask = np.ones((len(h),len(h)), img[0][0].dtype)
    for i in range(len(h)):
        for j in range(len(h)):
            mask[i][j]=(h[i] * h[j])
    localContrast = cv2.filter2D(img, 5,kernel=mask) 
    for i in range(len(localContrast)):
        for j in range(len(localContrast[0])):
            if localContrast[i][j] > math.pi / 2.75:
                localContrast[i][j] = math.pi / 2.75
    localContrast = cv2.subtract(img, localContrast)
    return cv2.multiply(localContrast, localContrast)


def Exposedness(img):
    sigma = 0.25
    average = 0.5
    exposedness = np.zeros(img.shape,img[0][0].dtype)
    for i in range(len(img)):
        for j in range(len(img[0])):
            value = math.exp(-1.0 * math.pow(img[i, j] - average, 2.0) / (2 * math.pow(sigma, 2.0)))
            exposedness[i][j] = value
    return exposedness

##############################################################
##############################################################

def filterMask(img):
    h = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]
    mask = np.zeros((len(h), len(h)), img[0][1].dtype)
    for i in range(len(h)):
        for j in range(len(h)):
            mask[i][j] = h[i] * h[j]
    return mask


def buildGaussianPyramid(img, level):
    gaussPyr =[]
    mask = filterMask(img)
    tmp = cv2.filter2D(img, -1, mask)
    gaussPyr.append(tmp.copy())
    tmpImg = img.copy()
    for i in range(1,level):
        cv2.resize(tmpImg, (0, 0), tmpImg, 0.5, 0.5, cv2.INTER_LINEAR) 
        tmp = cv2.filter2D(tmpImg,-1,mask)
        gaussPyr.append(tmp.copy())
    return gaussPyr


def buildLaplacianPyramid(img, level):
    lapPyr = []  
    lapPyr.append(img.copy())
    tmpImg = img.copy()
    tmpPyr = img.copy()
    for i in range(1,level):
        cv2.resize(tmpImg, (0, 0), tmpImg, 0.5, 0.5, cv2.INTER_LINEAR)
        lapPyr.append(tmpImg.copy())
    for i in range(level - 1):
        cv2.resize(lapPyr[i + 1], (len(lapPyr[i][0]), len(lapPyr[i])), tmpPyr, 0, 0, cv2.INTER_LINEAR)
        cv2.subtract(lapPyr[i], tmpPyr)
    return lapPyr


def reconstructLaplacianPyramid(pyramid): #5 levels are combined to one channel
    level = len(pyramid)
    for i in range(level - 1,0):
        tmpPyr = cv2.resize(pyramid[i], (len(pyramid[0][0]),len(pyramid[0])),fx= 0,fy= 0,interpolation=cv2.INTER_LINEAR)
        pyramid[i - 1] = cv2.add(pyramid[i - 1], tmpPyr)
    return pyramid[0]


def fuseTwoImages(w1, img1, w2, img2, level):
    weight1 = buildGaussianPyramid(w1, level)
    weight2 = buildGaussianPyramid(w2, level)
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    bgr = cv2.split(img1)
    bCnl1 = buildLaplacianPyramid(bgr[0], level)
    gCnl1 = buildLaplacianPyramid(bgr[1], level)
    rCnl1 = buildLaplacianPyramid(bgr[2], level)
    bgr = []
    bgr = cv2.split(img2)
    bCnl2 = buildLaplacianPyramid(bgr[0], level)
    gCnl2 = buildLaplacianPyramid(bgr[1], level)
    rCnl2 = buildLaplacianPyramid(bgr[2], level)
    bCnl=[]
    gCnl=[]
    rCnl=[]

    for i in range(level):
        cn = cv2.add(cv2.multiply(bCnl1[i], weight1[i]), cv2.multiply(bCnl2[i], weight2[i]))
        bCnl.append(cn.copy())
        cn = cv2.add(cv2.multiply(gCnl1[i], weight1[i]), cv2.multiply(gCnl2[i], weight2[i]))
        gCnl.append(cn.copy())
        cn = cv2.add(cv2.multiply(rCnl1[i], weight1[i]), cv2.multiply(rCnl2[i], weight2[i]))
        rCnl.append(cn.copy())
    bChannel = reconstructLaplacianPyramid(bCnl)
    gChannel = reconstructLaplacianPyramid(gCnl)
    rChannel = reconstructLaplacianPyramid(rCnl)
    fusion = cv2.merge((bChannel, gChannel, rChannel))
    return fusion

##############################################################
##############################################################


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix
def simplest_cb(img, percent):
    if percent <= 0:
        percent = 5
    img = np.float32(img)
    halfPercent = percent/200.0
    channels = cv2.split(img)
    results = []
    for channel in channels:
        assert len(channel.shape) == 2
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        flat = np.sort(flat)
        lowVal = flat[int(math.floor(flat.shape[0] * halfPercent))]
        topVal = flat[int(math.ceil(flat.shape[0] * (1.0 - halfPercent)))]
        channel = apply_threshold(channel, lowVal, topVal)
        normalized=cv2.normalize(channel,channel.copy(),0.0,255.0,cv2.NORM_MINMAX)
        channel= np.uint8(normalized)
        results.append(channel)

    return cv2.merge(results)

################################################################################
################################################################################

def enhance(image, level,check):
    if check == 1:
      img1 = simplest_cb(image, 5)
    else :
      img1 = image
    img1 = np.uint8(img1)
    LabIm1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab);
    L1 = cv2.extractChannel(LabIm1, 0);
    # Apply     
    result = applyCLAHE(LabIm1, L1)
    img2 = result[0]
    L2 = result[1]
    w1 = calWeight(img1, L1)
    w2 = calWeight(img2, L2)
    sumW = cv2.add(w1, w2)
    w1 = cv2.divide(w1, sumW)
    w2 = cv2.divide(w2, sumW)
    return fuseTwoImages(w1, img1, w2, img2, level)  


def applyCLAHE(img, L):
    clahe = cv2.createCLAHE(clipLimit=2.0)
    L2 = clahe.apply(L)
    lab = cv2.split(img)
    LabIm2 = cv2.merge((L2, lab[1], lab[2]))
    img2 = cv2.cvtColor(LabIm2, cv2.COLOR_Lab2BGR)
    result = []
    result.append(img2)
    result.append(L2)
    return result


def calWeight(img, L):
    L = np.float32(np.divide(L, (255.0)))
    WL = np.float32(LaplacianContrast(L)) # Check this line
    WC = np.float32(LocalContrast(L))
    WS = np.float32(Saliency(img))
    WE = np.float32(Exposedness(L))
    weight = WL.copy()
    weight = np.add(weight, WC)
    weight = np.add(weight, WS)
    weight = np.add(weight, WE)
    return weight
  
def enhanceColorCorrect(path,location):
  level = 5
  start = time.time()
  image = cv2.imread(path)
  fusion = enhance(image, 5,1)
  #print(time.time()-start)
  fusion= np.uint8(fusion)

  cv2.imwrite(location, fusion)
  cv2.imshow("Enhanced and Color Corrected", fusion)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
def ColorCorrect(path,location):
  level = 5
  image = cv2.imread(path)
  fusion = simplest_cb(image, 5)
  #print(time.time()-start)
  fusion= np.uint8(fusion)
  cv2.imwrite(location, fusion)
  cv2.imshow("Color Corrected", fusion)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def enhanceOnly(path,location):
  level = 5
  image = cv2.imread(path)
  fusion = enhance(image, 5, 0)
  fusion= np.uint8(fusion)

  cv2.imwrite(location, fusion)
  cv2.imshow("Enhanced",fusion)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
#########################################################################
#########################################################################
#enhanceColorCorrect("green_water_2.jpg","fusion.png")
#enhanceOnly(path,"fusion1.png")
#ColorCorrect("green_water_2.jpg","fusion2.png")