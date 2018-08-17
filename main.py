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
# https://pysource.com/2018/01/31/object-detection-using-hsv-color-space-opencv-3-4-with-python-3-tutorial-9/
# https://stackoverflow.com/questions/26932891/detect-touching-overlapping-circles-ellipses-with-opencv-and-python
# https://www.pyimagesearch.com/2015/11/02/watershed-opencv/#comment-421187
# https://docs.opencv.org/3.4/d2/dbd/tutorial_distance_transform.html
# https://github.com/muratkrty/specularity-removal
# https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
# https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/

import cv2
import imutils
import numpy as np
import time
import os

import table

imgInput = cv2.imread(".\captures\capture_005.jpg")
#imgInput = cv2.imread(".\captures\lines.png")
cv2.imshow("Image Input", imgInput)
table.extract_tapis(imgInput)


cv2.waitKey(0)
cv2.destroyAllWindows()
