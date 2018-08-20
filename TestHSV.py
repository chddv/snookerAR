import cv2
import numpy as np
import specularity as spc
 
def nothing(x):
    pass
 
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 600,600)
 
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.createTrackbar("Hough - min", "Trackbars", 1, 255, nothing)
cv2.createTrackbar("Hough - max", "Trackbars", 10, 255, nothing)


imgInput = cv2.imread("./captures/testhsv1.jpg")
#imgInput = cv2.imread("./captures/capture_001.jpg")

# retirrer le tappis
imgHSV =  cv2.cvtColor(imgInput, cv2.COLOR_BGR2HSV)
imgGreenThreshold = cv2.inRange(imgHSV, (45,0,0), (75,255,255))
imgThresholdInv = cv2.bitwise_not(imgGreenThreshold)
imgThresholdColor = cv2.cvtColor(imgGreenThreshold, cv2.COLOR_GRAY2BGR)
imgDiff = cv2.add(imgInput, imgThresholdColor)
imgDiffHSV =  cv2.cvtColor(imgDiff, cv2.COLOR_BGR2HSV)
imgGray = imgDiffHSV[:, :, 2]
cv2.imshow("imgGray", imgGray)
cv2.imshow("imgGreenThreshold", imgGreenThreshold)

r_img = m_img = np.array(imgGray)
rimg = spc.derive_m(imgInput, r_img)
s_img = spc.derive_saturation(imgInput, rimg)
spec_mask = spc.check_pixel_specularity(rimg, s_img)
cv2.imshow("spec_mask", spec_mask)
enlarged_spec = spc.enlarge_specularity(spec_mask)
radius = 12 
telea = cv2.inpaint(imgInput, enlarged_spec, radius, cv2.INPAINT_TELEA)
ns = cv2.inpaint(imgInput, enlarged_spec, radius, cv2.INPAINT_NS)
cv2.imshow("telea", telea)
cv2.imshow("ns", ns)

#imgBlurred = cv2.GaussianBlur(imgGray, (3, 3), 0)
#ret, imgThresh = cv2.threshold(imgBlurred,  250, 255, cv2.THRESH_BINARY_INV)
#kernel = np.ones((3,3), np.uint8)

#dist = cv2.distanceTransform(imgGreenThreshold, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
#dist = cv2.normalize(dist, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
#cv2.imshow("dist", dist)

#kernel = np.ones((3,3), np.uint8)
#morph = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel)
#morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
#th = cv2.adaptiveThreshold(imgDilate, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2 )

#kernel = np.ones((1,1), np.uint8)
#morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
#cv2.imshow("morph", morph)

#kernel = np.ones((3,3), np.uint8)
#imgErode = cv2.erode(imgDiffHSV[:, :, 2], kernel, iterations=1)
#cv2.imshow("imgErode", imgErode)




#kernel = np.ones((3,3), np.uint8)
#imgDilate = cv2.morphologyEx(imgInput, cv2.MORPH_CLOSE, kernel)
#kernel = np.ones((3,3), np.uint8)
#imgErode = cv2.erode(imgInput, kernel, iterations=1)
#imgDilate = cv2.dilate(imgErode, kernel, iterations=1)

while True:
    """
    hsv = cv2.cvtColor(imgInput, cv2.COLOR_BGR2HSV)

    
 
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    h_min = cv2.getTrackbarPos("Hough - min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hough - max", "Trackbars")
 
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
 
    result = cv2.bitwise_and(imgInput, imgInput, mask=mask)

    imgGray = cv2.cvtColor(imgDilate, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(imgGray, cv2.HOUGH_GRADIENT, 1, 2, param1=100, param2=50, minRadius=h_min, maxRadius=h_max  )
    if(not (circles is None) > 0):
        circles = np.uint16(np.around(circles))
        for c in circles[0,:]: 
            cv2.circle(result, (c[0], c[1]), c[2], (255,0,255), 1)

 
    cv2.imshow("frame", imgDilate)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)
    """
    key = cv2.waitKey(1)
    if key == 27:
        break
 
cv2.destroyAllWindows()