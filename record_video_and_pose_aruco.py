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


def aruco_single_pose_estimation(frame, aruco_dict, intrinsics, distortion):

    annotated_frame = copy.deepcopy(frame)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(annotated_frame, aruco_dict, parameters=parameters)

    rvecs = []
    tvecs = []
    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i],
                                                                           MARKER_LENGTH, intrinsics,
                                                                           distortion)

            # draw detected markers corners and borders
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners)

            # drawing axis of aruco
            cv2.drawFrameAxes(annotated_frame, intrinsics, distortion, rvec, tvec, length=.2, thickness=3)

            rvecs.append(rvec)
            tvecs.append(tvec)

    annotated_frame = aruco_display(corners, ids, rejected_img_points, annotated_frame, rvecs, tvecs)

    return annotated_frame, rvecs, tvecs


def aruco_board_pose_estimation(frame, aruco_dict, intrinsics, distortion):
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markers_w = 5  # Number of markers in the X direction.
    markers_h = 7  # Number of markers in the y direction.
    marker_length = 0.04  # length of aruco marker (m)
    marker_separation = 0.01  # separation between markers (m)

    # creat an aruco Board (The ids in ascending order starting on 0)
    grid_board = cv2.aruco.GridBoard((markers_w, markers_h),
                                     marker_length,
                                     marker_separation,
                                     aruco_dict)

    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    rvec = None
    tvec = None

    # if at least one marker is detected
    if len(corners) > 0:

        # estimate pose of board
        success, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, grid_board, intrinsics, distortion, rvec, tvec)

        if success:
            # draw detected markers corners
            cv2.aruco.drawDetectedMarkers(frame, corners)
            # drawing axis of aruco marker board
            # cv2.aruco.drawAxis(frame, intrinsics, distortion, rvec, tvec, 0.05)
            cv2.drawFrameAxes(frame, intrinsics, distortion, rvec, tvec, 0.05)

            # adding as text the distance from camera to aruco marker (z coord of tvec)
            cv2.putText(frame, str(int(tvec[2])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)
    else:
        # if no markers detected, placing text that board wasn't found
        cv2.putText(frame, 'no board detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

    return frame, rvec, tvec


def save_frame(img, frame_num, intrinsics, distortion):
    # undistort
    dst = cv2.undistort(img, intrinsics, distortion, None, intrinsics)

    # undistorted
    cv2.imwrite('{}/images/{:08d}.png'.format(save_folder, frame_num), img)  # save_folder, i
    # distorted
    cv2.imwrite('{}/undistorted/{:08d}.png'.format(save_folder, frame_num), dst)  # save_folder, i


if __name__ == '__main__':
    BOARD = True
    project_path = Path(__file__).parent.resolve()
    # folder will be structured as follows:
    # assets/type/folder/images

    type = 'aruco'  # random / phantom / EM_tracker_calib / tests
    folder = 'aruco_test'

    save_folder = f'{project_path}/assets/{type}/{folder}'

    MARKER_LENGTH = 30  # mm
    RECORD_ALL = False
    FRAME_RATE = 1
    # CREATING FOLDERS WHERE TO SAVE FRAMES AND TRACKING INFO
    if not os.path.isdir(f'{save_folder}/images'):
        os.makedirs(f'{save_folder}/images')
        os.makedirs(f'{save_folder}/undistorted')

    intrinsics = np.loadtxt(f'{project_path}/calibration/mac_calibration/intrinsics.txt')
    distortion = np.loadtxt(f'{project_path}/calibration/mac_calibration/distortion.txt')

    ar = cv2.aruco.DICT_4X4_50  # aruco dictionary we will use
    aruco_dict = cv2.aruco.getPredefinedDictionary(ar)  # dictionary of markers provided

    cap = cv2.VideoCapture(0)

    rvecs_all = []
    tvecs_all = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if BOARD:
            output, rvecs, tvecs = aruco_board_pose_estimation(frame, aruco_dict, intrinsics, distortion)
        else:
            output, rvecs, tvecs = aruco_single_pose_estimation(frame, aruco_dict, intrinsics, distortion)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            save_frame(frame, frame_num, intrinsics, distortion)
            rvecs_all.append(rvecs)
            tvecs_all.append(tvecs)

        if RECORD_ALL:
            if frame_num % FRAME_RATE == 0:
                if np.isnan(rvecs).all() or np.isnan(tvecs).all():
                    continue
                else:
                    save_frame(frame, frame_num, intrinsics, distortion)
                    rvecs_all.append(rvecs)
                    tvecs_all.append(tvecs)

        frame_num += 1
        if key == ord("q"):
            break

    rvecs_all = np.array(rvecs_all).squeeze()  # .mean(axis=0)
    tvecs_all = np.array(tvecs_all).squeeze()  # .mean(axis=0)

    # SAVE RT VECTORS AND TIMES
    np.save(f'{save_folder}/rvecs', np.array(rvecs_all))
    np.save(f'{save_folder}/tvecs', np.array(tvecs_all))

cv2.destroyAllWindows()
cap.release()
