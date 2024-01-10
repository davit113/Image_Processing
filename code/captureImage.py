import numpy as np
import cv2
import time

"""  """
def  captureImage(): 
    #raspberry pi camera captures 2592x1944 picture. So I think scale factor should be 0.25 for better performance, it will give us 648x486 image.
    #r.n. my laptop camera resolution is ~480p. so I set scale factor 1.
    SCALEFACTOR = 1 

    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap  = cv2.VideoCapture(0)   #selects default videocapture device 


    succes, frame =  cap.read()
    if  not succes: 
        print("something went wrong")  #we should implement better error/exeption handling later
        exit()
    else:
        time.sleep(2)    #to capture better image
        _ , frame = cap.read()
        smallFrame = cv2.resize(frame, (0,0), fx=SCALEFACTOR, fy=SCALEFACTOR)

    width = int(cap.get(3))
    height = int(cap.get(4))
    cv2.imshow('magic', frame)
        



    cap.release()      #definattely better releaser needed X button does not reliases resourses
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return frame



