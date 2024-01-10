""" this file includes filtering functions for images  """

import numpy as np
import cv2


""" all filtering of BGR camptured image """
def filterBGR(img):
    img = cv2.bilateralFilter(img, 15, 75, 75)
    img = cv2.GaussianBlur(img,(11,11),0)
    pass


""" all filtering of HSV image(converted from BGR, so it runs after filter BGR(or it is plan anyway)) """
def filterHSV(img):

    pass

""" doubleThreshold :), or mayby later double adaptive threshhold)) """
def doubleThreshold(img, val1, val2, adaptive=False):

    if not adaptive:
        _, mask1 = cv2.threshold(img, val1, 255, cv2.THRESH_BINARY)
        if 0 <=val2 and val2 <255:
            _, mask2 = cv2.threshold(img, val2, 255, cv2.THRESH_BINARY_INV)
        else: return mask1
        return  cv2.bitwise_and(mask1, mask2)
    else:
        # ????
        mask1 = thresholdAdaptive(img, val1)
        return  mask1

def thresholdAdaptive(img,val):
    MAX_VALUE = 255
    BLOCK_SIZE = 11 #size of neigbor area
    CONSTANT = 2
    return cv2.adaptiveThreshold(img, MAX_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, CONSTANT)


""" rans once per cycle, on binary images to eliminate samll gaps, or single points """
def filterBinary(thrash):
    # morphology cleaning
    structuringElement = np.ones((5,5), np.uint8) 
    thrash = cv2.erode(thrash, structuringElement, iterations=1)
    return  cv2.dilate(thrash, structuringElement, iterations=2)


