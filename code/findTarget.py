""" 
In HSV colorspace
    yeloww hue=50~70 saturation=210~250
    blue   hue=185~200 saturation=175~235
 """

import numpy as np
import cv2
import time
import captureImage

SCALEFACTOR = 1.0


def findTarget():
    img = cv2.imread('./assets/1m_2.2.jpg')
    # img = captureImage.captureImage()
    img = cv2.resize(img, (0,0), fx=SCALEFACTOR, fy=SCALEFACTOR)
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("imgg", imgGrey)
    _, thrash = cv2.threshold(imgGrey, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("imggg", thrash)
    contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    counter =0
    for contour in contours:
        counter+=1
        approx = cv2.approxPolyDP(contour, 0.001* cv2.arcLength(contour, True), True)
        if(cv2.contourArea(contour) < 1000):
            continue
        else: 
            print(cv2.contourArea(contour),'---/n')

        var= str(cv2.contourArea(contour))

        cv2.drawContours(img, [approx], 0, (0, 0, 0), 1)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        if len(approx) == 3:
            cv2.putText(img, "triangle" +var, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.1, (0, 0, 0),1)
        else:
            cv2.putText(img, "circle"+ var, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),1)

    print (counter)
    cv2.imshow("shapes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
