import cv2
import numpy as np 

acolor = np.uint8([[[0,0,255]]])
hsv_color = cv2.cvtColor(acolor, cv2.COLOR_BGR2HSV)
print(hsv_color)