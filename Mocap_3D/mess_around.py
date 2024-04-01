import cv2
from cvzone.ColorModule import ColorFinder
import cvzone
import time
import pickle
import numpy as np
from scipy import stats

class KalmanFilter:
    timestep = 1/10.2054668528855 # FPS
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    print(kalman_filter.processNoiseCov)
    kalman_filter.processNoiseCov = np.array([[0.25*timestep**4, 0, 0, 0], [0, 0.25*timestep**4, 0, 0], [0, 0, 0.5*timestep**2, 0], [0, 0, 0, 0.5*timestep**2]], dtype=np.float32) * 1 # set noise from acceleration
    kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.01 # set noise from pixel measurements

    def predict(self, x_position, y_position):
        measured = np.array([[np.float32(x_position)], [np.float32(y_position)]])
        self.kalman_filter.correct(measured)
        predicted = self.kalman_filter.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

cap = cv2.VideoCapture(1)
cap.set(3, 1920)
cap.set(4, 1080)

cv_file = cv2.FileStorage()
cv_file.open('stereo_calibration.xml', cv2.FileStorage_READ)
mtx2 = cv_file.getNode('matrix_2').mat()
M2 = cv_file.getNode('newmatrix_2').mat()
dist2 = cv_file.getNode('distCoeffs_2').mat()
P2 = cv_file.getNode('projection_2').mat()

# Marker colors
findColor = ColorFinder(False)
markerColors = {'neon_green' : {'hmin': 40, 'smin': 40, 'vmin': 50, 'hmax': 80, 'smax': 255, 'vmax': 255},
                'neon_pink' : {'hmin': 160, 'smin': 100, 'vmin': 150, 'hmax': 180, 'smax': 255, 'vmax': 255},
                'blue' : {'hmin': 100, 'smin': 60, 'vmin': 110, 'hmax': 120, 'smax': 200, 'vmax': 255},
                'yellow' : {'hmin': 25, 'smin': 40, 'vmin': 170, 'hmax': 40, 'smax': 120, 'vmax': 255}}


# Save video
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 

original = cv2.VideoWriter('messy_filename.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         6, size) 

# Delay before beinning recording
WAIT_TIME = 1000
record_length = 30
cv2.waitKey(WAIT_TIME)
start = time.time()
initial_time = time.time()

# Save data
timeVector = []
green_markers = []
kalman_tracking = []

kf = KalmanFilter()

while (start < initial_time + record_length):


    success, img = cap.read()

    img = cv2.flip(img, 0)
    img = cv2.undistort(img, mtx2, dist2, M2)

    imgColor_green, mask_green = findColor.update(img, markerColors['neon_pink'])
    imgContour_green, contours_green = cvzone.findContours(img, mask_green, minArea=200)

    contours_green.sort(key=lambda x: x['center'][1])

    if len(contours_green) > 0:
        green_markers.append(contours_green[0]['center'])
        print(green_markers)

    # imgStack = cvzone.stackImages([mask_green, mask_pink, imgContour_green, imgContour_pink], 2, 0.5)

    if len(green_markers) > 1:
        cv2.circle(img, green_markers[-1], 15, (0, 20, 220), 4)
        predicted = kf.predict(green_markers[-1][0], green_markers[-1][1])
        cv2.circle(img, (predicted[0], predicted[1]), 15, (20, 220, 0), 4)
        kalman_tracking.append([predicted[0], predicted[1]])

    cv2.imshow("Image", img)

    original.write(img) 

    end = time.time()
    totalTime = end - start
    start = time.time()
    fps = 1 / totalTime
    timeVector.append(start-initial_time)
    print(f"FPS: {fps}")
    cv2.waitKey(1)

print(f"Time Period Mean: {np.mean(np.array(timeVector)**-1)}")
print(f"Time Period SEM: {stats.sem(np.array(timeVector)**-1)}")

with open('messy_info_2D', 'wb') as handle:
    pickle.dump({'kalman_tracking': kalman_tracking, 'green_markers': green_markers, 'timeVector' : timeVector}, handle, protocol=pickle.HIGHEST_PROTOCOL)


cap.release() 
original.release()
cv2.destroyAllWindows() 