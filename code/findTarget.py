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

TOLERANCE_FOR_CIRCLE = 0.90
TOLERANCE_FOR_TRIANGLE = 0.90


""" on 2592x1944 picture, with scale=0.25

    1. distance=1m circleAreaBlue=[5565,5605,5600,5550] circleAreaYelow=[5933,5946] triangleArea=[3310,3226]

    2  distance=2m circleAreaBlue=[1498,1509,1515,1480,1514] , circleAreaYelow~= [1332(bad!),1394(bad!)], triangleArea=[814,811,823]
                                    should be around 1400,                                       
 """

import numpy as np
import cv2
import math

import code.filtering as myFiltr

from matplotlib import pyplot as plt

def cleanImage(img):
    myFiltr.filterBGR(img)
    # cv2.imshow("BGRCleaned", img)

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
    # maskBlueSat =  myFiltr.doubleThreshold(sat1, BLUE_COLORS_DOWN[1] ,BLUE_COLORS_UP[1]) # reusing maskYelowSat, becouse it has almos identical value

    # cv2.imshow("YH", maskYelowHue)
    # cv2.imshow("YS", maskYelowSat)
    # cv2.imshow("BH", maskBlueHue)
    # cv2.imshow("BS", maskBlueSat)

    final_yelow_mask_image = cv2.bitwise_and(maskYelowHue, maskYelowSat)
    final_blue_mask_image  = cv2.bitwise_and(maskBlueHue, maskYelowSat)
        
    yelBlueMask = cv2.bitwise_or(final_blue_mask_image, final_yelow_mask_image)
    return yelBlueMask
    return [final_yelow_mask_image, final_blue_mask_image]

def findTarget(img, cfg):  # cfg=[bool,bool]
    thrash = cleanImage(img)
    cv2.imshow("Cleaned", thrash)
    myFiltr.filterBinary(thrash)
    cv2.imshow("binaryCleaned", thrash)

    contours, _ = cv2.findContours(thrash, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS) 

    findSomething =False
    for contour in contours:
        conturArea= cv2.contourArea(contour)
        if(conturArea < 300):
            continue

        minCircleArea =  cv2.minEnclosingCircle(contour)[1]**2 * math.pi 
        minTriangleArea= cv2.minEnclosingTriangle(contour)[0] 

        x = contour.ravel()[0]
        y = contour.ravel()[1] - 5

        if minCircleArea * TOLERANCE_FOR_CIRCLE <  conturArea and cfg[1]:
            findSomething = True
            cv2.drawContours(img, [contour], 0, (0, 0, 0), 1)
            cv2.putText(img, f"CIR: {conturArea},dis={math.sqrt(5600/conturArea):.2f}M;", (x-80, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.33, (0, 0, 150),1)
        if minTriangleArea * TOLERANCE_FOR_TRIANGLE < conturArea and cfg[0]:
            findSomething = True
            cv2.drawContours(img, [contour], 0, (0, 0, 0), 1)
            cv2.putText(img, f"TR: {conturArea},dis={(math.sqrt(3300/conturArea)):.2f}M;", (x-80, y), cv2.FONT_HERSHEY_COMPLEX, 0.33, (0, 150, 0),1)

    if not  findSomething:
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

