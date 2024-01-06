import pyConfig as cfg

import numpy as np
import cv2
import functools
import threading
import calcDistance as dis
import captureImage as cap
import findTarget as ftar
import targetSize as tsize


from tkinter import *

def action(val: int):
    print(val)
    if val == 0:  cap.captureImage() 
    elif val == 1: ftar.findTarget()
    elif val == 2: tsize.targetSize()
    elif val ==3: dis.calcDistance()
    else: print("error!"); exit



def main():

    window = Tk()
    window.geometry("420x420")
    window.title("Image Processing")
    icon = PhotoImage(file='a3.png')
    window.iconphoto(True, icon)  

    btnArr = []

    def _():
        count = 0
        for bn in cfg.BUTTON_NAMES:
            print(count, '/n')
            param = count 
            btn = Button(window,
                        command=functools.partial(action, count),
                        text=bn,
                        font=(cfg.FONT_MAIN,20),
                        fg=cfg.FONT_COLOR_MAIN,
                        bg=cfg.BG_COLOR_MAIN,
                        activeforeground=cfg.FONT_COLOR_MAIN,
                        activebackground=cfg.BG_COLOR_MAIN
                        )
            btnArr.append(btn)
            count+=1

        for bn in btnArr:
            bn.pack()
    _()
    window.mainloop()



def vaitFun(cap,):
    print ("DASFAFSFAFFAFs")
    while True:
        print('--t')
        if  cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            


def testFoo():
    
    cap = cv2.VideoCapture(0)
    # wait_thread = threading.Thread(target=vaitFun, args=(cap,))
    # wait_thread.start()

    while True:
        ret, frame = cap.read()
        newFrame = ftar.findTarget(frame)
        cv2.imshow('farme', newFrame)
        if  cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    
    
def foo():
    wait_thread = threading.Thread(target=vaitFun, args=(cap,10))
    wait_thread.start()
    wait_thread.join()




if(__name__ == '__main__'):
    testFoo();    
    # foo()





