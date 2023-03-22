# conda -> test_env
# conda install -c conda-forge python=3.7.8
from pathlib import Path

## use x86_64 architecture channel(s)
# conda config --env --set subdir osx-64

## install python, numpy, etc. (add more packages here...)
# conda install python=3.7 numpy
import cv2
import os
import numpy as np
import glob


def obtain_calibration(img, CHESSBOARD_SIZE, criteria):

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # If found, add object points, image points (after refining them)
    if ret:
        # refine corners found of chessboard
        corners2 = cv2.cornerSubPix(gray, corners, CHESSBOARD_SIZE, (-1, -1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('frame', img)
        cv2.waitKey(1000)

        return ret, corners2
    else:
        return ret, False


def calibrate_live_stream(objp, criteria, CHESSBOARD_SIZE):
    first = True


    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # opening webcam and Checking webcam is opened correctly
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        k = cv2.waitKey(1)
        # get frame captured from webcam
        ret, img = cap.read()
        # display frame so user can see what it looks like
        cv2.imshow('frame', img)

        # getting frame size (only necessary once)
        if first:
            frame_size = img.shape[:-1]
            print(frame_size)
            first = False

        # CAPTURING FRAME
        # wait for capturing image- if user presses c, the current frame on display will be
        # used for calibration
        if k == ord('c'):
            ret, corners2 = obtain_calibration(img, CHESSBOARD_SIZE, criteria)

            # If found, add object points and image points (after refining them)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners2)

        # wait for ESC key or quit key to exit and terminate feed.
        if k == 27 or k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    return objpoints, imgpoints, frame_size


def calibrate_images(image_path, objp, criteria, CHESSBOARD_SIZE):
    first = True
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(f'{image_path}/*')

    for image in images:

        # LOAD and display IMAGE
        img = cv2.imread(image)
        cv2.imshow('img.jpg', img)
        #cv2.waitKey(1)

        # obtain frame size (only first time)
        if first:
            frame_size = img.shape[:-1]
            first = False

        # for each frame obtain corners of checkerbpard
        ret, corners2 = obtain_calibration(img, CHESSBOARD_SIZE, criteria)

        # If found, add object points and image points (after refining them)
        if ret:
            print('corners found')
            objpoints.append(objp)
            imgpoints.append(corners2)

        # after all images have been used, destroy all windows
        cv2.destroyAllWindows()

    return objpoints, imgpoints, frame_size


def calibrate_camera(CHESSBOARD_SIZE, VIDEO_TYPE, image_path=''):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    # find chessboard corners from either images or live stream
    if VIDEO_TYPE == 'stream':
        objpoints, imgpoints, frame_size = calibrate_live_stream(objp, criteria, CHESSBOARD_SIZE)
    elif VIDEO_TYPE == 'images':
        objpoints, imgpoints, frame_size = calibrate_images(image_path, objp, criteria, CHESSBOARD_SIZE)

    ret, mtx, dist, camera_rvecs, camera_tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    return ret, mtx, dist, camera_rvecs, camera_tvecs


def main():
    # where calibration matrices will be stored
    calibration_name = 'phone_calibration'
    project_path = Path(__file__).parent.resolve()
    if not os.path.isdir(f'{project_path}/{calibration_name}'):
        os.makedirs(f'{project_path}/{calibration_name}')

    # CAMERA CALIBRATION SETTINGS
    rows = 14  # inner number of corners along row
    cols = 10  # inner number of corners along col
    CHESSBOARD_SIZE = (rows, cols)

    # TYPE - are you recording a live stream or set of images
    VIDEO_TYPE = 'images'  # stream / images
    IMAGE_PATH = f'{project_path}/phone_calib_images'

    # calibrate camera
    ret, intrinsics, distortion, camera_rvecs, camera_tvecs = calibrate_camera(CHESSBOARD_SIZE,VIDEO_TYPE, image_path=IMAGE_PATH)

    print('camrea calibrated', ret)
    print('camera matrix:', intrinsics)
    print('distortion params:', distortion)
    print('rotation vecs:', camera_rvecs)
    print('translation vecs:', camera_tvecs)

    np.savetxt(f'{project_path}/{calibration_name}/intrinsics.txt', intrinsics)
    np.savetxt(f'{project_path}/{calibration_name}/distortion.txt', distortion)
    np.save(f'{project_path}/{calibration_name}/camera_rvecs', camera_rvecs)
    np.save(f'{project_path}/{calibration_name}/camera_tvecs', camera_tvecs)


if __name__ == "__main__":
    main()