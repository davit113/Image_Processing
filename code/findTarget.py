""" 
In HSV colorspace
    yeloww hue=50~70 saturation=210~250 // in cv2 hue is about = 25~33 and saturation= 200~250
    blue   hue=185~200 saturation=175~235  // in cv2 hue is about = 92~100 and saturation= 175~240
 """
""" 
    note: 
        cv2.COLOR_BGR2HSV does convertion wheere H- max value is 180.
 """
"""                  H    S    V """
YELOW_COLORS_DONW = [20, 150, 20]
YELOW_COLORS_UP   = [35, 255, 255]

BLUE_COLORS_DOWN = [90, 140, 20]
BLUE_COLORS_UP   = [110, 255, 255]


import numpy as np
import cv2
import time


import code.filtering as myFiltr

from cv2 import imshow as show
from matplotlib import pyplot as plt

def cleanImage(img):
    SCALEFACTOR = 1
    img = cv2.resize(img, (0,0), fx=SCALEFACTOR, fy=SCALEFACTOR)
    myFiltr.filterBGR(img)

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    myFiltr.filterHSV(hsv)
    hue1,sat1,v = cv2.split(hsv)

    """ double threshholding for yelow color """
    """ double threshholding for yelow saturation"""
    """ double threshholding for blue color """
    """ double threshholding for blue saturation"""
    maskYelowHue = myFiltr.doubleThreshold(hue1, YELOW_COLORS_DONW[0] ,YELOW_COLORS_UP[0])
    maskYelowSat = myFiltr.doubleThreshold(sat1, YELOW_COLORS_DONW[1] ,YELOW_COLORS_UP[1])
    maskBlueHue =  myFiltr.doubleThreshold(hue1, BLUE_COLORS_DOWN[0] ,BLUE_COLORS_UP[0])
    maskBlueSat =  myFiltr.doubleThreshold(sat1, BLUE_COLORS_DOWN[1] ,BLUE_COLORS_UP[1])

    final_yelow_mask_image = cv2.bitwise_and(maskYelowHue, maskYelowSat)
    final_blue_mask_image  = cv2.bitwise_and(maskBlueHue, maskBlueSat)
        
    yelBlueMask = cv2.bitwise_or(final_blue_mask_image, final_yelow_mask_image)
    return yelBlueMask
    return [final_yelow_mask_image, final_blue_mask_image]


def findTarget(img, mode=0):
    thrash = cleanImage(img)
    myFiltr.filterBinary(thrash)
    contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    counter = 0
    for contour in contours:
        counter+=1
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        if(cv2.contourArea(contour) < 500):
            continue

        counter+=1
        var= str(cv2.contourArea(contour))
        cv2.drawContours(img, [approx], 0, (0, 0, 0), 1)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        if len(approx) == 3  and (mode == 0 or mode==2):
            cv2.putText(img, "triangle" +var, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),3)
        elif len(approx) > 30  and (mode == 0 or mode==1):
            cv2.putText(img, "circle"+ var, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),3)

    if counter == 0:
        cv2.putText(img, "nothing here...", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0),10)
    return img


def testFoo(img =0):
    img = cv2.imread('code/tt.jpg', cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

    kernal = np.ones((5,5), np.uint8)

    dilation = cv2.dilate(mask, kernal, iterations=2)
    erosion = cv2.erode(mask, kernal, iterations=1)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal)
    th = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernal)

    titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'mg', 'th']
    images = [img, mask, dilation, erosion, opening, closing, mg, th]

    for i in range(8):
        plt.subplot(2, 4, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

def findAndDrawTriangle(img):
    findTarget(img,1)

def findAndDrawCircle(img):
    findTarget(img,2)



if(__name__ == "__main__"):
    findTarget()
    pass


