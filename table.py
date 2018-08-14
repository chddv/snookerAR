import cv2
import imutils
import numpy as np

# Adaptive threshold levels
BKG_THRESH = 65
CARD_THRESH = 30

def contourToRect(_contour):
    # Attention: pts est un countour (list de list de point) [[[2]]]
    # initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = np.sum(_contour, axis = 2)
	rect[0] = _contour[np.argmin(s)]
	rect[2] = _contour[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(_contour, axis = 2)
	rect[1] = _contour[np.argmin(diff)]
	rect[3] = _contour[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect


def four_point_transform(_image, _contour):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = contourToRect(_contour)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(_image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

""""
Tapis = Vert
Red Ball = 1 point 
Yellow Ball = 2 points
Green Ball = 3 points
Brown Ball = 4 points
Blue Ball = 5 points
Pink Ball = 6 points
Black Ball = 7 points
"""
BALL_RED = 1
BALL_YELLOW = 2
BALL_GREEN = 3
BALL_BROWN = 4
BALL_BLUE = 5
BALL_PINK = 6
BALL_BLACK = 7
HSV_LOWER = 0
HSV_UPPER = 1

""" Method to get Green Limit of HSV
> green = np.uint8([[[0,255,0 ]]])
> hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
> print hsv_green
[[[ 60 255 255]]]
[H-10, 100,100] and [H+10, 255, 255] as lower bound and upper bound 
"""
TAPIS_HSV_LOWER_UPPER = [[45,0,0],[75,255,255]]

BALL_HSV_LOWER_UPPER = [[[6,4,17],[75,66,255]],
                    [[27,213,249],[24,244,255],],
                    [[20,0,0],[40,255,255]],
                    [[50,0,0],[70,255,255]],
                    [[2,0,0],[22,255,255]],
                    [[110,0,0],[130,255,255]],
                    [[157,0,0],[177,255,255]],
                    [[0,0,0],[2,255,255]]]
               

hsvImage = None
def onMouse(event, x, y, flag, param):
    if(event == cv2.EVENT_LBUTTONUP):
        pixel = hsvImage[y, x]
        print(pixel)

def extract_ball(_image, _color):
    # convert to hsv format
    hsvImg =  cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower = np.array(BALL_HSV_LOWER_UPPER[_color][HSV_LOWER]) #formule non respecté => permet de ramener aussi du vert plus foncé...
    upper = np.array(BALL_HSV_LOWER_UPPER[_color][HSV_UPPER])

    # threshold image to only green part
    imgThreshold = cv2.inRange(hsvImg, lower, upper)
    cv2.imshow("ball imgThreshold", imgThreshold)

def extract_tapis(_image):
    """ retourne la masque pour ne garder que le tapis sur l'image complette """

    """ *** 1er Etape: Determiner la zone du tapis """

    imgMul = cv2.multiply(_image, np.array([1.75])) # adjust exposure (depending of the source image) TODO: trouver un mmoyen de trouver la bonne valeur (1.75 ... )
    cv2.imshow("imgMul", imgMul)

    # convert to hsv format
    hsvImg =  cv2.cvtColor(imgMul, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_green = np.array(TAPIS_HSV_LOWER_UPPER[HSV_LOWER]) #formule non respecté => permet de ramener aussi du vert plus foncé...
    upper_green = np.array(TAPIS_HSV_LOWER_UPPER[HSV_UPPER])

    # threshold image to only green part
    imgThreshold = cv2.inRange(hsvImg, lower_green, upper_green)
    #cv2.imshow("imgThreshold", imgThreshold)

    imgThresholdInv = cv2.bitwise_not(imgThreshold)
    imgThresholdColor = cv2.cvtColor(imgThresholdInv, cv2.COLOR_GRAY2BGR)

    """ *** 2e Etape: Masquer tous le reste de l'image """

    print(imgThreshold.shape)
    print(imgThresholdColor.shape)
    print(_image.shape)
    #cv2.imshow("imgThresholdColor", imgThresholdColor)
    imgDiff = cv2.add(_image, imgThresholdColor)
    #cv2.imshow("imgDiff", imgDiff)

    """ *** 3e Etape: Retrouver les bordures de la tables """
    """     2 WAYS : 
            - Find contours (approxPoly) => TODO: ajouter recuperation les angle maxi mini (top left, top right, bottom left, bottom right)
            - TODO: use lines ()

    """

    imgGray = cv2.cvtColor(imgDiff, cv2.COLOR_BGR2GRAY)    
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)

    

    retval, imgThresh = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edged = cv2.Canny(imgThresh, 100, 200, apertureSize=3)

    kernel = np.ones((10,10), np.uint8)
    img_erosion = cv2.erode(imgThreshold, kernel, iterations=1)
    edged = cv2.dilate(img_erosion, kernel, iterations=1)

    dummyImage, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]

    # loop over our contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) #0.05 * peri
        global hsvImage
        hsvImage = four_point_transform(imgMul, approx)
        cv2.imshow("imageRes", hsvImage)
        cv2.setMouseCallback("imageRes", onMouse )
    extract_ball(hsvImage, 1)


    """
    lines = cv2.HoughLinesP(edged, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    for l in lines:
       x1,y1,x2,y2  = l[0]
       cv2.line(_image,(x1,y1),(x2,y2),(0,0,255),3)
    cv2.imshow("src _image", _image)
    """

    """
    # retirrer les artefact (bump / unbump pour garder le reste)
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((10,10), np.uint8)
    img_erosion = cv2.erode(imgThreshold, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    cv2.imshow("img_dilation", img_dilation)
    
    # keep the whole tapis WITH balls
    im2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    order_index = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
    #x,y,w,h = cv2.boundingRect(contours[0])
    #cv2.rectangle(_image,(x,y),(x+w,y+h),(0,255,0),2)

    epsilon = 0.01*cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[order_index[0]], epsilon, True)
    #cv2.drawContours(_image, approx,-1, (0,0,255), 3)

    #cv2.drawContours(_image, contours[order_index[0]],-1, (0,0,255), 3)
    #cv2.fillPoly(_image, pts =[contours[order_index[0]]], color=(255,255,255))
    cv2.fillPoly(_image, pts =[approx], color=(255,255,255))
    cv2.imshow("im2", _image)

    """


def preprocess_table(_image):
    """Preprocess source image , and return a threshold image
    """

    imgMul = cv2.multiply(_image, np.array([1.75])) # adjust exposure (depending of the source image) TODO: trouver un mmoyen de trouver la bonne valeur (1.75 ... )

    imgGray = cv2.cvtColor(imgMul, cv2.COLOR_BGR2GRAY)
    
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)
    cv2.imshow("imgBlur", imgBlur)

    #img_w, img_h = np.shape(_image)[:2]
    #bkg_level = imgGray[int(img_h/2)][int(img_w/2)]
    #thresh_level = bkg_level + BKG_THRESH
    #retval, imgThresh = cv2.threshold(imgBlur, thresh_level, 255, cv2.THRESH_BINARY)

    #imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 2)
    
    retval, imgThresh = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("imgThresh", imgThresh)

    edged = cv2.Canny(imgThresh, 100, 200, apertureSize=3)
    #edged = cv2.bitwise_not(edged)
    cv2.imshow("edged", edged) 

    """
    dummyImage, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:20]

    # loop over our contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        cv2.drawContours(_image, [c], -1, (0, 255, 0), 3)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        #if len(approx) == 4:
        #    screenCnt = approx
        #    break
    """
    #cv2.drawContours(_image, [screenCnt], -1, (0, 255, 0), 3)
    lines = cv2.HoughLinesP(edged, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    for l in lines:
       x1,y1,x2,y2  = l[0]
       cv2.line(_image,(x1,y1),(x2,y2),(0,0,255),3)


    cv2.imshow("_image with countour", _image) 
    """
    # find lines
    lines = cv2.HoughLines(edged, 1, np.pi/180.0, 100, np.array([]), 0, 0)
    a,b,c = lines.shape
    # filter lines by theta and compute average
    theta_min = 60 * np.pi / 180
    theta_max = 120 * np.pi / 180
    theta_avr = 0
    theta_deg = 0
    filtered_lines = []
    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        if(theta > theta_min and theta < theta_max) :
            filtered_lines.append(lines[i])
            theta_avr += theta
    if (len(filtered_lines) > 0) :
        theta_avr = theta_avr / len(filtered_lines)
        theta_deg = (theta_avr / np.pi * 180) - 90
    """


    return imgThresh

def find_table(_imgThresh, _imgSrc):
    """Uses contour to isolate the table
    """

    dummyImage, contours, hierarchy = cv2.findContours(_imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    order_index = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
    filtered_index = order_index[0:2]
    
    filtered_contours = []
    for i in filtered_index:
        filtered_contours.append(contours[i])

    debug_contours(_imgSrc, filtered_contours)

def debug_contours(_img, _cnts):
    if len(_cnts) > 0:
        cv2.drawContours(_img, _cnts, -1, (0,0,255), 1)
    cv2.imshow("contour", _img)    
