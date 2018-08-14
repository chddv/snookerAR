# Help from https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
# https://solarianprogrammer.com/2015/05/08/detect-red-circles-image-using-opencv/
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
# https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
# http://www.4answered.com/questions/view/d601e8/Detect-Snooker-Billiard-Balls-with-Opencv
# https://gist.github.com/cancan101/e19982815e4d3b2b3b95
# http://www.marcus-hebel.de/foto/ShiftN_Fonctions.html
# http://www.marcus-hebel.de/

import cv2
import imutils
import numpy as np
import time
import os

import table

imgInput = cv2.imread(".\captures\capture_005.jpg")
#imgInput = cv2.imread(".\captures\lines.png")
cv2.imshow("Image Input", imgInput)

#imgTableThresh = table.preprocess_table(imgInput) 
#table.find_table(imgTableThresh, imgInput)
table.extract_tapis(imgInput)

cv2.waitKey(0)
cv2.destroyAllWindows()
