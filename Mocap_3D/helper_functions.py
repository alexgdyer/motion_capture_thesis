import cvzone
from cvzone.ColorModule import ColorFinder
import cv2
import pickle
import time
import numpy as np

# Dictionary to store logged marker colors (edit to match)
markerColors = {'neon_green' : {'hmin': 40, 'smin': 40, 'vmin': 50, 'hmax': 80, 'smax': 255, 'vmax': 255},
                'neon_pink' : {'hmin': 160, 'smin': 100, 'vmin': 150, 'hmax': 180, 'smax': 255, 'vmax': 255},
                'blue' : {'hmin': 100, 'smin': 60, 'vmin': 110, 'hmax': 120, 'smax': 200, 'vmax': 255},
                'yellow' : {'hmin': 25, 'smin': 40, 'vmin': 120, 'hmax': 40, 'smax': 160, 'vmax': 255},
                'orange' : {'hmin': 13, 'smin': 60, 'vmin': 150, 'hmax': 40, 'smax': 180, 'vmax': 255}}

def findMarkers(img, color, minArea):
    findColor = ColorFinder(False)
    
    # Check input
    if (not markerColors.get(color)):
        print(f" Provided color is not logged. Logged marker colors are {list(markerColors.keys())}. Proceeding with neon_green")
        color = 'neon_green'

    # Extract the contours for the image
    img_color, mask = findColor.update(img, markerColors[color])
    img_contour, marker_contours = cvzone.findContours(img_color, mask, minArea=minArea)

    return mask, [x['center'] for x in marker_contours]


def triangulate_markers(img_left, img_right, markers_left, markers_right, projection_left, baseline, img_height, cameras_angle):

    # Sort markers by vertical location
    markers_left.sort(key=lambda x: x[1])
    markers_right.sort(key=lambda x: x[1])

    markers_coords = []

    # Iterate through the found marker contours and match
    left_index = 0
    right_index = 0
    while left_index < len(markers_left) and right_index < len(markers_right):

        # Check to see if tracked objects are the same by comparing vertical location
        # We check to see that the vertical displacement is greater than 5% of the image heights since the two cameras should only be horizontally displaced
        if (np.abs(markers_left[left_index][1] - markers_right[right_index][1]) > 0.2 * img_height):
            print(f"Markers are not the same")
            if markers_left[left_index][1] > markers_right[right_index][1]:
                left_index += 1
            else:
                right_index += 1
            
            continue
        
        # If the markers match, calculate the labratory space coordinates
        markers_coords.append(calculate_3d_point(img_left, img_right, markers_left[left_index], markers_right[right_index], projection_left, baseline, cameras_angle))

        left_index += 1
        right_index += 1

    return markers_coords


def calculate_3d_point(img_left, img_right, projPoint_left, projPoint_right, projMtx_left, baseline, cameras_angle):
    
    height_right, width_right, depth_right = img_right.shape
    height_left, width_left, depth_left = img_left.shape

    f_pixel = (width_right * 0.5) / np.tan(110 * 0.5 * np.pi/180)

    disparity =  projPoint_right[0] - projPoint_left[0]
    x = baseline * (projPoint_left[0] - width_left//2) / disparity
    y = -baseline * (projPoint_left[1] - height_left//2) / disparity
    z = baseline * 2.54 * f_pixel / disparity

    return [x, y, z]


def get_contours_video(img, masks, minArea):
    '''
    img: the image from the camera upon which the bounding boxes will be drawn
    colorsList: list of strings corresponding to the marker colors dictionary
    minArea: minimum area for an object to be considered a marker
    '''
 
    if (len(masks) > 0):
        imgContours = img.copy()
        for mask in masks:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if minArea < area:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                    cv2.drawContours(imgContours, cnt, -1, (255, 0, 0), 3)
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.putText(imgContours, str(area), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    cx, cy = x + (w // 2), y + (h // 2)
                    cv2.rectangle(imgContours, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.circle(imgContours, (x + (w // 2), y + (h // 2)), 5, (255, 0, 0), cv2.FILLED)

        return imgContours

    print("No masks provided")
    return img


def conduct_trial(baseline, record_video, display_video, record_length, minArea, markerColors, cameras_angle, size, frame_rate):
    '''
    baseline: distance between cameras (in cm)
    record_video: title for whether trial should be recorded, does not record if None
    display_video: display the recording from the left camera
    record_length: time to record in seconds
    size: image size of the cameras
    '''

    # Get calibration parameters
    with open('stereo_calibration', 'rb') as f:
        calibration_info = pickle.load(f)

    mtx1 = calibration_info['matrix_1']
    mtx2 = calibration_info['matrix_2']
    CM1 = calibration_info['newmatrix_1']
    CM2 = calibration_info['newmatrix_2']
    dist1 = calibration_info['distCoeffs_1']
    dist2 = calibration_info['distCoeffs_2']
    P1 = calibration_info['projection_1']
    P2 = calibration_info['projection_2']
    mapx_left = calibration_info['mapx_left']
    mapy_left = calibration_info['mapy_left']
    mapx_right = calibration_info['mapx_right']
    mapy_right = calibration_info['mapy_right']

    # Get video capture from two web cameras
    FRAME_HEIGHT = size[0]
    FRAME_WIDTH = size[1]
    cap_right = cv2.VideoCapture(0)
    cap_right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap_right.set(cv2.CAP_PROP_FPS, 30)
    cap_left = cv2.VideoCapture(1)
    cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap_left.set(cv2.CAP_PROP_FPS, 30)

    if (record_video is not None):
        result = cv2.VideoWriter(record_video,  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        frame_rate, (FRAME_HEIGHT, FRAME_WIDTH)) 

    timeVector = []
    markerVector = []

    # Delay before beinning recording
    WAIT_TIME = 500
    cv2.waitKey(WAIT_TIME)
    start = time.time()
    initial_time = time.time()

    while (start < initial_time + record_length):

        # Read current images and undistort
        succes_right, img_right = cap_right.read()
        succes_left, img_left = cap_left.read()
        img_right = cv2.undistort(img_right, mtx2, dist2, CM2)
        img_left = cv2.undistort(img_left, mtx2, dist2, CM2)

        
        # Get masks
        masks = []
        markers_coord_dict = {}
        for color in markerColors:
            mask, markers_left = findMarkers(img_left, color, minArea)
            masks.append(mask)
            _, markers_right = findMarkers(img_right, color, minArea)
            markers_coords = triangulate_markers(img_left, img_right, markers_left, markers_right, P1, baseline, FRAME_HEIGHT, cameras_angle)
            markers_coord_dict[color] = markers_coords

            # print(f"left: {markers_green_left}")
            # print(f"right: {markers_green_right}")

        markerVector.append(markers_coord_dict)

        if (display_video or record_video is not None):

            # Draw bounding boxes around found markers
            img_left_markers = get_contours_video(img_left, masks, minArea)            
            if (display_video):
                cv2.imshow("Image", img_left_markers)
            if (record_video is not None):
                result.write(img_left_markers) 

        cv2.waitKey(1)

        end = time.time()
        totalTime = end - start
        start = time.time()
        fps = 1 / totalTime
        timeVector.append(start-initial_time)
        # print("FPS: ", fps)
        # print(f"Time since starting {end-initial_time}")

    cap_left.release() 
    cap_right.release() 

    # Finish recording
    if (record_video is not None):
        result.release()

    cv2.destroyAllWindows() 

    return timeVector, markerVector


def record_trial(record_length, size, frame_rate):
    '''
    baseline: distance between cameras (in cm)
    record_video: title for whether trial should be recorded, does not record if None
    display_video: display the recording from the left camera
    record_length: time to record in seconds
    size: image size of the cameras
    '''

    # Get video capture from two web cameras

    FRAME_HEIGHT = size[0]
    FRAME_WIDTH = size[1]
    cap_right = cv2.VideoCapture(0)
    cap_right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap_right.set(cv2.CAP_PROP_FPS, 30)
    cap_left = cv2.VideoCapture(1)
    cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap_left.set(cv2.CAP_PROP_FPS, 30)

    video_left = cv2.VideoWriter('Video_Files/video_left.avi',  
                    cv2.VideoWriter_fourcc(*'MJPG'), 
                    frame_rate, (FRAME_HEIGHT, FRAME_WIDTH)) 
    
    video_right = cv2.VideoWriter('Video_Files/video_right.avi',  
                    cv2.VideoWriter_fourcc(*'MJPG'), 
                    frame_rate, (FRAME_HEIGHT, FRAME_WIDTH)) 

    timeVector = []
    markerVector = []

    # Delay before beinning recording

    WAIT_TIME = 5000
    cv2.waitKey(WAIT_TIME)
    prev_time = time.time() - 1/frame_rate
    start = time.time()
    initial_time = time.time()
    images_right = []
    images_left = []

    print("Beginning recording")

    while (start < initial_time + record_length):

        # Ensure that the recording is not taking place faster than the cameras' refresh rate
        # while ((time.time() - prev_time) - 1/frame_rate < 0):
        #     time.sleep(0.01)
        prev_time = time.time()

        # Read current images and add them to the queue
        succes_right, img_right = cap_right.read()
        succes_left, img_left = cap_left.read()
        images_right.append(img_right)
        images_left.append(img_left)

        # Store information related to timing
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        # print("FPS: ", fps)
        # print(f"Time since starting {end-initial_time}")
        start = time.time()
        timeVector.append(start-initial_time)


    cap_left.release() 
    cap_right.release() 
    cv2.destroyAllWindows() 
    print(f"Finished recording with a total of {len(images_left)} frames")
    print(1/np.mean(np.diff(timeVector)))
    print(np.std(1/np.diff(timeVector)))

    print("Beginning storing...")
    record_begin = time.time()

    for img_left, img_right in zip(images_left, images_right):
        video_left.write(img_left)
        video_right.write(img_right)
        # cv2.imshow("Image_l", img_left)
        # cv2.imshow("Image_r", img_right)
        # cv2.waitKey(1)

    print(f"Done recording after {time.time() - record_begin} seconds")

    video_left.release()
    video_right.release()

    

    return timeVector


def analyze_trial(baseline, record_video, display_video, minArea, markerColors, cameras_angle, size, frame_rate):
    '''
    baseline: distance between cameras (in cm)
    record_video: title for whether trial should be recorded, does not record if None
    display_video: display the recording from the left camera
    record_length: time to record in seconds
    size: image size of the cameras
    '''

    # Get calibration parameters
    with open('stereo_calibration', 'rb') as f:
        calibration_info = pickle.load(f)

    mtx1 = calibration_info['matrix_1']
    mtx2 = calibration_info['matrix_2']
    CM1 = calibration_info['newmatrix_1']
    CM2 = calibration_info['newmatrix_2']
    dist1 = calibration_info['distCoeffs_1']
    dist2 = calibration_info['distCoeffs_2']
    P1 = calibration_info['projection_1']
    P2 = calibration_info['projection_2']
    mapx_left = calibration_info['mapx_left']
    mapy_left = calibration_info['mapy_left']
    mapx_right = calibration_info['mapx_right']
    mapy_right = calibration_info['mapy_right']

    # Get video capture from two web cameras
    FRAME_HEIGHT = size[0]
    FRAME_WIDTH = size[1]
    cap_right = cv2.VideoCapture('Video_Files/video_right.avi')
    cap_right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap_left = cv2.VideoCapture('Video_Files/video_left.avi')
    cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

    if (record_video is not None):
        result = cv2.VideoWriter(record_video,  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        frame_rate, (FRAME_HEIGHT, FRAME_WIDTH)) 

    markerVector = []
    timeVectorAnalysis = []

    # Delay before beinning recording
    WAIT_TIME = 2000
    cv2.waitKey(WAIT_TIME)
    start = time.time()
    initial_time = time.time()

    print('Beginning analysis...')

    while True:

        # Read current images and undistort
        succes_right, img_right = cap_right.read()
        succes_left, img_left = cap_left.read()

        if img_right is None or img_left is None:
            break
        
        img_right = cv2.flip(img_right, 0)
        img_left = cv2.flip(img_left, 0)

        img_right = cv2.undistort(img_right, mtx2, dist2, CM1)
        img_left = cv2.undistort(img_left, mtx2, dist2, CM2)

        # Get masks
        masks = []
        markers_coord_dict = {}
        for color in markerColors:
            mask, markers_left = findMarkers(img_left, color, minArea)
            masks.append(mask)
            _, markers_right = findMarkers(img_right, color, minArea)
            markers_coords = triangulate_markers(img_left, img_right, markers_left, markers_right, P1, baseline, FRAME_HEIGHT, cameras_angle)
            markers_coord_dict[color] = markers_coords

            # print(f"left: {markers_green_left}")
            # print(f"right: {markers_green_right}")

        markerVector.append(markers_coord_dict)

        if (display_video or record_video is not None):

            # Draw bounding boxes around found markers
            img_left_markers = get_contours_video(img_left, masks, minArea)            
            if (display_video):
                cv2.imshow("Image", img_left_markers)
            if (record_video is not None):
                result.write(img_left_markers) 

        cv2.waitKey(1)

        end = time.time()
        totalTime = end - start
        start = time.time()
        timeVectorAnalysis.append(start-initial_time)
        # fps = 1 / totalTime
        # print("FPS: ", fps)
        # print(f"Time since starting {end-initial_time}")

    cap_left.release() 
    cap_right.release() 

    # Finish recording
    if (record_video is not None):
        result.release()

    cv2.destroyAllWindows() 

    print('Finished analysis')
    print(1/np.mean(np.diff(timeVectorAnalysis)))
    print(np.std(1/np.diff(timeVectorAnalysis)))

    return np.array(markerVector)