import copy

import cv2
import numpy as np
from pathlib import Path
import os


def aruco_display(corners, ids, rejected, image, rvecs, tvecs):
    if len(corners) > 0:
        # ids of aruco markers detected
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 3)
            # print("ArUco marker ID: {}".format(markerID))

        avg_z = np.array(tvecs).mean(axis=0).squeeze()

        # WRITING DISTANCE
        cv2.putText(image, str(int(avg_z[2])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

    return image


def aruco_single_pose_estimation(frame, aruco_dict, marker_length, intrinsics, distortion):
    """ This function detects a single aruco marker in the video stream

    This function will detect the pose of an aruco marker relative to the camera.
    It will return the pose as-well as the frame with annotations including- borders
    around the marker detected, the bottom left corner, axes of the aruco board, and
    the z distance will be shown on the top left of the image.

    Args:
        frame: frame we're recording and want to detect marker on
        aruco_dict: aruco dictionary we're using
        marker_length: length of marker in mm
        intrinsics: intrinsics of cam used to record
        distortion: distortion coeffs of cam used to record

    Returns:
        annotated_frame (NXM): frame with annotations of aruco board positions
        rvec (): rotations of aruco marker relative to camera
        tvec (): translations of aruco marker relative to camera
    """

    # copy of frame to be annotated
    annotated_frame = copy.deepcopy(frame)
    # detecting aruco markers
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(annotated_frame, aruco_dict, parameters=parameters)

    rvecs = []
    tvecs = []
    if len(corners) > 0:
        # going through each detected aruco marker
        for i in range(0, len(ids)):
            # estimating pose of aruco marker
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i],
                                                                           marker_length, intrinsics,
                                                                           distortion)

            # draw detected markers corners and borders
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners)

            # drawing axis of aruco
            cv2.drawFrameAxes(annotated_frame, intrinsics, distortion, rvec, tvec, length=10, thickness=3)

            rvecs.append(rvec)
            tvecs.append(tvec)

    # adding distance to camera and aruco number
    annotated_frame = aruco_display(corners, ids, rejected_img_points, annotated_frame, rvecs, tvecs)

    return annotated_frame, rvecs, tvecs


def aruco_board_pose_estimation(frame, aruco_dict, grid_board, intrinsics, distortion):
    """function for detecting aruco board pose

    This function will detect the pose of an aruco board relative to the camera.
    It will return the poses as-well as the frame with annotations including- borders
    around each marker detected, the bottom left corner, axes of the aruco board, and
    the z distance will be shown on the top left of the image.

    Args:
        frame (NXM): frame with aruco baord in it
        aruco_dict (obj): aruco dictionary we will use (eg. cv2.aruco.DICT_4X4_50)
        grid_board (obj): grid board defined with cv2.aruco.GridBoard((markers_w, markers_h),
                                                                      marker_length,
                                                                      marker_separation,
                                                                      aruco_dict)
        intrinsics: intrinsics matrix of camera we're using to record
        distortion: distortion coeffs of camera we're using to record

    Returns:
        annotated_frame (NXM): frame with annotations of aruco board positions
        rvec (): rotations of aruco board relative to camera
        tvec (): translations of aruco board relative to camera
    """

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # making copy of frame for annotation
    annotated_frame = copy.deepcopy(frame)

    # detector = cv2.aruco.ArucoDetector()
    parameters = cv2.aruco.DetectorParameters()
    # corners, ids, rejected_img_points = detector.detectMarkers(frame)
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    rvec = None
    tvec = None

    # if at least one marker is detected
    if len(corners) > 0:

        # estimate pose of board
        success, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, grid_board, intrinsics, distortion, rvec, tvec)

        if success:
            # draw detected markers corners on frame
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners)
            # drawing axis of aruco marker board
            #cv2.drawFrameAxes(annotated_frame, intrinsics, distortion, rvec, tvec, 0.05)
            cv2.drawFrameAxes(annotated_frame, intrinsics, distortion, rvec, tvec, 25, thickness=5 )
            # adding as text the distance from camera to aruco marker (z coord of tvec)
            cv2.putText(annotated_frame, str(tvec[2][0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        # if no markers detected, placing text that board wasn't found
        cv2.putText(annotated_frame, 'no board detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    return annotated_frame, rvec, tvec


def save_frame(img, save_folder, frame_num, intrinsics, distortion):
    """ Function for saving frame

    This function will save both original frame under <save_folder>/images and
    the undistorted frame under <save_folder>/images

    Args:
        img (NXM): frame we want to save
        save_folder (str): folder directory where images will be saved
        frame_num (int): frame number- this is for the name of the img when stored
        intrinsics : intrinsics matrix of camera used to record
        distortion: distortion coeffs of camera used to record

    Returns:
        saves images (original and distorted) under save folder
    """

    # undistort
    dst = cv2.undistort(img, intrinsics, distortion, None, intrinsics)

    # undistorted
    cv2.imwrite('{}/images/{:08d}.png'.format(save_folder, frame_num), img)  # save_folder, i
    # distorted
    cv2.imwrite('{}/undistorted/{:08d}.png'.format(save_folder, frame_num), dst)  # save_folder, i


def record_video_and_poses(save_folder,
                           intrinsics,
                           distortion,

                           cap_port=0,
                           board=True,

                           record_all_frames=True,
                           frame_rate=1,

                           aruco_dict=cv2.aruco.DICT_4X4_50,
                           # if aruco board is used
                           markers_w=5,  # Number of markers in the X direction.
                           markers_h=7,  # Number of markers in the y direction.
                           marker_length=16,  # length of aruco marker (mm)
                           marker_separation=5  # separation between markers (mm)
                           ):
    """Function that records and save video frames from camera stream and records aruco poses

    This function will record a images from the camera port you specify.
    If you specify record_all_frames to True, all frames will be recorded at the specified frame rate.
    If you select record_all_frames as False, you'll have to press 'c' to capture each frame that you want

    For each frame you attempt to capture, the aruCo board or markers will be detected and the pose of
    the board estimated if it is found.

    Args:
        save_folder (str): folder where you want the images and the camera poses to be saved
        intrinsics (ndarray): intrinsics matrix of camera you'll be recording with
        distortion (ndarray): distortion matrix of camera you'll be recording with
        cap_port (int): port of camera used for cv2.VideoCapture()- if using your computer's camera this is typically 0
        board (bool): if True, the poses will be estimated based on aruco board.
                      If false, poses will be estimated based on single aruco marker
        record_all_frames (bool): If true this will record all the frames at set frame rate.
                                  If false, you are able to choose which frames to capture by using the key 'c'
        frame_rate (int): rate at which frames are recordedd (note that this is only applicable if record_all_frames=True)

        aruco_dict (obj): aruco dictionary that was used to generate the markers you will be recording (eg. cv2.aruco.DICT_4X4_50)
        markers_w (int): how many aruco markers along width of board
        markers_h (int): how many aruco markers along height of board
        marker_length (float): length of markers (mm)
        marker_separation (float): separation between markers (mm)

    Returns:
        The original images will be saved under <save_folder>/images/*.png
        The undistorted images will be saved under <save_folder>/undistorted/*.png
        The rotation poses of the aruco relative to the camera will be saved under <save_folder>/rvecs.npy
        The translation poses of the aruco relative to the camera will be saved under <save_folder>/rtvecs.npy

    """

    # CREATING FOLDERS WHERE TO SAVE FRAMES AND TRACKING INFO
    if not os.path.isdir(f'{save_folder}/images'):
        os.makedirs(f'{save_folder}/images')
        os.makedirs(f'{save_folder}/undistorted')

    # loading aruco dictionary that was printed for detection
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)  # dictionary of markers provided

    # initialising videocapture. Note you can change the port to be webcam or something else
    cap = cv2.VideoCapture(cap_port)

    # where all tracking data will be stored
    rvecs_all = []
    tvecs_all = []

    frame_num = 0

    # recording frames and poses
    while True:
        # extracting current frame
        ret, frame = cap.read()
        if not ret:
            break

        # obtaining poses-
        if board:

            # creat an aruco Board (The ids in ascending order starting on 0)
            grid_board = cv2.aruco.GridBoard((markers_w, markers_h),
                                             marker_length,
                                             marker_separation,
                                             aruco_dict)
            # obtaining pose estimation of board
            output, rvecs, tvecs = aruco_board_pose_estimation(frame, aruco_dict, grid_board, intrinsics, distortion)
        else:
            # if single marker is used
            output, rvecs, tvecs = aruco_single_pose_estimation(frame, aruco_dict, marker_length, intrinsics, distortion)

        # showing labelled frame
        cv2.imshow('Estimated Pose', output)

        # recording frames
        key = cv2.waitKey(1) & 0xFF
        if record_all_frames:
            # if user selected to record all frames, we record all frames within
            # frame rate specified
            if frame_num % frame_rate == 0:
                # if no tracking data, don't record this frame
                if np.isnan(rvecs).all() or np.isnan(tvecs).all():
                    continue
                else:
                    save_frame(frame, save_folder, frame_num, intrinsics, distortion)
                    rvecs_all.append(rvecs)
                    tvecs_all.append(tvecs)
        else:
            # if user didn't select record_all_frames, they will press
            # the key 'c' for capturing a frame
            if key == ord('c'):
                if np.isnan(rvecs).all() or np.isnan(tvecs).all():
                    continue
                save_frame(frame, save_folder, frame_num, intrinsics, distortion)
                rvecs_all.append(rvecs)
                tvecs_all.append(tvecs)

        frame_num += 1
        # if user presses 'q' they quit the recording
        if key == ord("q"):
            break

    # saving poses recorded of aruco board/marker
    rvecs_all = np.array(rvecs_all).squeeze()  # .mean(axis=0)
    tvecs_all = np.array(tvecs_all).squeeze()  # .mean(axis=0)

    # SAVE RT VECTORS AND TIMES
    np.save(f'{save_folder}/rvecs', np.array(rvecs_all))
    np.save(f'{save_folder}/tvecs', np.array(tvecs_all))

    cv2.destroyAllWindows()
    cap.release()


def main():
    BOARD = True
    project_path = Path(__file__).parent.resolve()

    # where to save data
    # folder will be structured as follows: assets/type/folder/images
    type = 'aruco'  # random / phantom / EM_tracker_calib / tests
    folder = 'sofa'
    save_folder = f'{project_path}/assets/{type}/{folder}'

    RECORD_ALL = False
    FRAME_RATE = 1

    ar = cv2.aruco.DICT_4X4_50  # aruco dictionary we will use

    intrinsics = np.loadtxt(f'{project_path}/calibration/mac_calibration/intrinsics.txt')
    distortion = np.loadtxt(f'{project_path}/calibration/mac_calibration/distortion.txt')

    record_video_and_poses(save_folder,
                           intrinsics,
                           distortion,

                           cap_port=0,
                           board=BOARD,

                           record_all_frames=RECORD_ALL,
                           frame_rate=FRAME_RATE,

                           aruco_dict=ar,
                           # if aruco board is used
                           markers_w=5,  # Number of markers in the X direction.
                           markers_h=7,  # Number of markers in the y direction.
                           marker_length=16,  # length of aruco marker (mm)
                           marker_separation=5  # separation between markers (mm)
                           )


if __name__ == '__main__':
    main()
