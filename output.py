import numpy as np
import cv2
import time
import random

width = 50
system = np.identity(width)

cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)

def display():
    #imshow is 0-1 for floats or 0-255 for ints
    cv2.imshow("image",system)
    cv2.waitKey(1)
    # time.sleep(1)


for i in range(10):
    x,y=(random.randint(0,width-1),random.randint(0,width-1 ))
    system[x][y] = 0.3
    display()


cv2.waitKey()

cv2.destroyAllWindows()