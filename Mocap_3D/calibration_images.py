import cv2
import cvzone
import time
import os

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)


images_to_capture = 20
delay_before_start = 5000 # in ms
image_delay = 1 # in seconds
last_time = time.time()
counter = 0
cv2.waitKey(delay_before_start)

while counter < images_to_capture:

    # Read in the images
    succes1, img = cap.read()
    succes2, img2 = cap2.read()
    img = cv2.flip(img, 0)
    img2 = cv2.flip(img2, 0)


    # Display current frame
    imgStack = cvzone.stackImages([img, img2], 2, 0.5)
    cv2.imshow("Image", imgStack)

    # Take a picture every few seconds per the specified delay
    if time.time() > last_time + image_delay:
        last_time = time.time()

        cv2.imwrite('images/images_left/image_left_' + str(counter) + '.png', cv2.resize(src=img, dsize=(1920, 1080), interpolation=cv2.INTER_AREA))
        cv2.imwrite('images/images_right/image_right_' + str(counter) + '.png', cv2.resize(src=img2, dsize=(1920, 1080), interpolation=cv2.INTER_AREA))
        print(f"Image {counter} captured successfully")
        counter += 1
        
    cv2.waitKey(1)

