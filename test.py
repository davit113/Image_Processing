import code.findTarget
import cv2

# img = cv2.imread('assets/1m_1.1.jpg')
# img = cv2.imread('assets/1m_1.2.jpg')
img = cv2.imread('assets/1m_2.2.jpg')
# img = cv2.imread('assets/1m_2.1.jpg')
# img = cv2.imread('assets/2m_1.1.jpg')
# img = cv2.imread('assets/2m_1.2.jpg')
# img = cv2.imread('assets/2m_2.1.jpg')
# img = cv2.imread('assets/2m_2.2.jpg')
img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
img = code.findTarget.findTarget(img)
cv2.imshow("foo", img)
cv2.waitKey(100000)
