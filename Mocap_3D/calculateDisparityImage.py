import cv2
import cvzone
import time
import os
import numpy as np

cap_right = cv2.VideoCapture(0)
cap_left = cv2.VideoCapture(1)

cv_file = cv2.FileStorage()
cv_file.open('stereo_calibration.xml', cv2.FileStorage_READ)
mtx1 = cv_file.getNode('matrix_1').mat()
mtx2 = cv_file.getNode('matrix_2').mat()
M1 = cv_file.getNode('newmatrix_1').mat()
M2 = cv_file.getNode('newmatrix_2').mat()
dist1 = cv_file.getNode('distCoeffs_1').mat()
dist2 = cv_file.getNode('distCoeffs_2').mat()
P1 = cv_file.getNode('projection_1').mat()
P2 = cv_file.getNode('projection_2').mat()

counter = 0
images_to_capture = 40

image_delay = 2 # in seconds
last_time = time.time()
cv2.waitKey(2000)
while counter < images_to_capture:

    # Read in the images
    succes1, img_right = cap_right.read()
    succes2, img_left = cap_left.read()

    img_right = cv2.undistort(img_right, mtx1, dist1, M1)
    img_left = cv2.undistort(img_left, mtx2, dist2, M2)

    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

    # Display current frame
    imgStack = cvzone.stackImages([img_left, img_right], 2, 0.5)
    cv2.imshow("Image", imgStack)

    disparityMap = np.zeros(np.shape(img_left))

    for i in len(img_left):
        for j in len(img_right):
            
        
    cv2.waitKey(5000)

