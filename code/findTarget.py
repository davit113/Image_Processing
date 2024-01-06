""" 
In HSV colorspace
    yeloww hue=50~70 saturation=210~250 // in cv2 hue is about = 25~33 and saturation= 200~250
    blue   hue=185~200 saturation=175~235  //// in cv2 hue is about = 92~100 and saturation= 175~240
 """

""" 
    note: 
        cv2.COLOR_BGR2HSV does convertion wheere H- max value is 180.
 """
import numpy as np
import cv2
import time
import captureImage

SCALEFACTOR = 1.0

from matplotlib import pyplot as plt



def cleanImage(img):
    SCALEFACTOR = 1
    # img = cv2.imread('code/new_test_img.jpg')
    img = cv2.resize(img, (0,0), fx=SCALEFACTOR, fy=SCALEFACTOR)

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    hue1,sat1,v = cv2.split(hsv)

    """ double threshholding for yelow color """
    _, maskHue1 = cv2.threshold(hue1, 25, 255, cv2.THRESH_BINARY)
    _, maskHue2 = cv2.threshold(hue1, 33, 255, cv2.THRESH_BINARY_INV)
    maskYelowHue= cv2.bitwise_and(maskHue1, maskHue2)

    """ double threshholding for yelow saturation"""
    _, maskSat1 = cv2.threshold(sat1, 200, 255, cv2.THRESH_BINARY)
    _, maskSat2 = cv2.threshold(sat1, 245, 255, cv2.THRESH_BINARY_INV) #probably not even needed here.
    maskYelowSat = cv2.bitwise_and(maskSat1, maskSat2)
    

    final_yelow_mask_image = cv2.bitwise_and(maskYelowHue, maskYelowSat)
    # cv2.imshow('finalYelow', final_yelow_mask_image)

    """ double threshholding for blue color """
    _, maskHue1 = cv2.threshold(hue1, 92, 255, cv2.THRESH_BINARY)
    _, maskHue2 = cv2.threshold(hue1, 100, 255, cv2.THRESH_BINARY_INV)
    maskBlueHue= cv2.bitwise_and(maskHue1, maskHue2)

    """ double threshholding for blue saturation"""
    _, maskSat1 = cv2.threshold(sat1, 175, 255, cv2.THRESH_BINARY)
    _, maskSat2 = cv2.threshold(sat1, 240, 255, cv2.THRESH_BINARY_INV) #probably not even needed here.
    maskBlueSat = cv2.bitwise_and(maskSat1, maskSat2)
    final_blue_mask_image = cv2.bitwise_and(maskBlueHue, maskBlueSat)

    # cv2.imshow('finalBlue', final_blue_mask_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        
    yelBlueMask = cv2.bitwise_or(final_blue_mask_image, final_yelow_mask_image)
    return yelBlueMask
    return [final_yelow_mask_image, final_blue_mask_image]


def findTarget(img):
    # img = captureImage.captureImage()
    # img = cv2.imread('code/new_test_img.jpg')
    # img = cv2.resize(img, (0,0), fx=SCALEFACTOR, fy=SCALEFACTOR)

    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thrash = cv2.threshold(imgGrey, 100, 255, cv2.THRESH_BINARY)
    thrash = cleanImage(img)

    # morphology cleaning
    structuringElement = np.ones((5,5), np.uint8) 
    thrash = cv2.erode(thrash, structuringElement, iterations=1)
    thrash = cv2.dilate(thrash, structuringElement, iterations=2)
    #####################

    # cv2.imshow("img", thrash)
    contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    counter = 0
    for contour in contours:
        counter+=1
        approx = cv2.approxPolyDP(contour, 0.001* cv2.arcLength(contour, True), True)
        if(cv2.contourArea(contour) < 1000):
            continue

        counter+=1
        var= str(cv2.contourArea(contour))
        cv2.drawContours(img, [approx], 0, (0, 0, 0), 1)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        if len(approx) == 3:
            cv2.putText(img, "triangle" +var, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.1, (0, 0, 0),1)
        else:
            cv2.putText(img, "circle"+ var, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),1)

    if counter == 0:
        cv2.putText(img, "nothing here...", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0),10)
    return img


def testFoo(img=0):
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

def if_main():
    if(__name__ == "__main__"):
        findTarget()
        pass
if_main()