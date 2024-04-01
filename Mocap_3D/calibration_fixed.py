# Code adapted from Temuge Batpurev
# https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

import cv2 as cv
import glob
import numpy as np
import pickle
 
def calibrate_camera(images_folder):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
 
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.0001)
 
    rows = 6 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 1 #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
 
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
 
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            # cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            # cv.imshow('img', frame)
            # k = cv.waitKey(500)
 
            objpoints.append(objp)
            imgpoints.append(corners)
 
 
 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)
 
    return mtx, dist, ret

def stereo_calibrate(mtx1, dist1, mtx2, dist2, images_folder_1, images_folder_2):
    #read the synched frames
    c1_images_names = sorted(glob.glob(images_folder_1))
    c2_images_names = sorted(glob.glob(images_folder_2))
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    print(len(c1_images))
    print(len(c2_images))

    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 500, 0.00001)
 
    rows = 6 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 1 #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
    counter = 0
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
            cv.imshow(f'img', frame1)
            cv.imwrite('images/images_left/image_left_calibration.png', cv.resize(src=frame1, dsize=(1920, 1080), interpolation=cv.INTER_AREA))
 
            cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
            cv.imshow(f'img2', frame2)
            cv.imwrite('images/images_left/image_right_calibration.png', cv.resize(src=frame2, dsize=(1920, 1080), interpolation=cv.INTER_AREA))
            k = cv.waitKey(1000)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

        counter += 1
 
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    R1, R2, P1, P2, Q, rot, trans = cv.stereoRectify(CM1, dist1, CM2, dist2, (width, height), R, T)
    
    print(ret)

    mapx_left, mapy_left = cv.initUndistortRectifyMap(CM1, dist1, R1, P1, (width, height), cv.CV_32FC1)
    
    mapx_right, mapy_right = cv.initUndistortRectifyMap(CM2, dist2, R2, P2, (width, height), cv.CV_32FC1)

    return CM1, CM2, P1, P2, dist1, dist2, rot, trans, mapx_left, mapy_left, mapx_right, mapy_right

images_folder_1 = "images/images_left/*"
images_folder_2 = "images/images_right/*"

mtx1, dist1, ret1 = calibrate_camera(images_folder_1)
mtx2, dist2, ret2 = calibrate_camera(images_folder_2)

CM1, CM2, P1, P2, dist1, dist2, rot, trans, mapx_left, mapy_left, mapx_right, mapy_right = stereo_calibrate(mtx1, dist1, mtx2, dist2, images_folder_1, images_folder_2)

print(P1)
print(P2)

# print("Saving parameters!")

# cv_file = cv.FileStorage('stereo_calibration.xml', cv.FILE_STORAGE_WRITE)
# cv_file.write('matrix_1', mtx1)
# cv_file.write('matrix_2', mtx2)
# cv_file.write('newmatrix_2', CM1)
# cv_file.write('newmatrix_1', CM2)
# cv_file.write('distCoeffs_1', dist1)
# cv_file.write('distCoeffs_2', dist2)
# cv_file.write('projection_1',P1)
# cv_file.write('projection_2',P2)
# cv_file.write('mapx_left',mapx_left)
# cv_file.write('mapy_left',mapy_left)
# cv_file.write('mapx_right',mapx_right)
# cv_file.write('mapy_right',mapy_right)
# cv_file.release()

with open('stereo_calibration', 'wb') as handle:
    pickle.dump({'matrix_1' : mtx1, 'matrix_2' : mtx2, 'newmatrix_1' : CM1, 'newmatrix_2' : CM2,
                 'distCoeffs_1' : dist1, 'distCoeffs_2' : dist2, 'projection_1' : P1, 'projection_2' : P2,
                 'mapx_left' : mapx_left, 'mapy_left' : mapy_left, 'mapx_right' : mapx_right, 'mapy_right' : mapy_right}, 
                 handle, protocol=pickle.HIGHEST_PROTOCOL)