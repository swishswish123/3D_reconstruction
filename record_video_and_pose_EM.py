from sksurgerynditracker.nditracker import NDITracker
import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from reconstruction_utils.utils import extrinsic_matrix_to_vecs


def record_RT_vecs(save_folder, TRACKER,intrinsics, distortion,capture_all, max_frames, frame_rate):

    # define a video capture object
    vid = cv2.VideoCapture(1)

    i = 0

    # Where we will save 4x4 matrices and times
    tvecs_all = []
    rvecs_all = []
    times = []

    points = []
    vectors = []

    # point at origin of tracker
    original_point = np.array([0,0,0,1])
    unit_vec = np.array([1,0,0,0])

    while(True):
        k = cv2.waitKey(33)

        # Capture the video frame by frame
        ret, frame = vid.read()

        # UNDISTORT FRAME
        dst = cv2.undistort(frame, intrinsics, distortion, None, intrinsics)
        cv2.imshow(f'images', dst)

        # current frame time
        if i ==0:
            print("recording....")
            start_time = time.time()
            time_diff = 0
        else:
            time_diff =  time.time()-start_time

        # capuring current frame (either when user presses 'c' or when they want to capture all frames
        if k == ord('c') or capture_all: # or i%FRAME_RATE==0:
            # when capturing all frames, skip those that aren't according to the frame rate
            if capture_all and not i%frame_rate==0:
                continue
            # extracting frame of tracking data from the NDI device.
            port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()

            # SAVE FRAME
            cv2.imwrite('{}/images/{:08d}.png'.format(save_folder,i), frame)#save_folder, i
            # SAVE UNDISTORTED FRAME
            cv2.imwrite('{}/undistorted/{:08d}.png'.format(save_folder,i), dst)

            # SAVE 4X4 MATRICES
            rvecs, tvecs = extrinsic_matrix_to_vecs(tracking[0])
            rvecs_all.append(rvecs)
            tvecs_all.append(tvecs)

            times.append(time_diff)
            print(tracking[0])

            # PLOT POINT for visualising later
            point = tracking[0]@original_point
            vector = tracking[0] @ unit_vec

            points.append(point)
            vectors.append(vector)

        # the 'q' button is set as the quitting button 
        if cv2.waitKey(1) & 0xFF == ord('q') or i==max_frames:
            # After the loop release the cap object
            vid.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
            break

        i+=1
    
    return tvecs_all, rvecs_all, times, np.array(points), np.array(vectors)


def record_data_EM(save_folder,calibration_path, capture_all=False,  frame_rate=1, max_frames=np.inf):
    """
    function that records video of images given initialised tracker

    Note that the aurora and the endoscope need to be properly connected to the computer you are running
    the code from.

    This function will record from the endoscope port when connected.
    If capture_all is true, the function will record all frames but if not it will capture only frames when you press
    the key 'c' on your computer

    The function will then save the tracking data and the images both distorted and undistorted in the given save folder

    Args:
        save_folder: folder where data will be stored
        calibration_path: path where calibration intrinsics.txt and distortion.txt files can be found.
        capture_all: whether all frames should be captured (True) or only those when you press 'c' (False)
        max_frames: max number of frames to be captured

    Returns:
        inside the save_folder, you will have two folders and 3 files:
        images/*.png will contain all the captured original images
        undistorted/*.png will contain all the captured images but undistorted
        rvecs.pny will contain the rotation vectors of the images captured
        tvecs.npy will contain the translation vectors of the images captured
        times.npy will contain all the timestamps
    """
    # where intrinsics and distortion are located
    intrinsics = np.loadtxt(f'{calibration_path}/intrinsics.txt')
    distortion = np.loadtxt(f'{calibration_path}/distortion.txt')

    # CREATING FOLDERS WHERE TO SAVE FRAMES AND TRACKING INFO
    if not os.path.isdir(f'{save_folder}/images'):
        os.makedirs(f'{save_folder}/images')
        os.makedirs(f'{save_folder}/undistorted')

    # initialising aurora EM tracker
    print('init params')
    # init params
    SETTINGS = {
        "tracker type": "aurora",
    }
    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()
    print('finished init')

    # recordinng frames and Rt vectors
    tvecs, rvecs, times, points, vectors = record_RT_vecs(save_folder, TRACKER,intrinsics, distortion,capture_all, max_frames, frame_rate)

    # visualising tracked camera positions and rotations
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((5, 1, 5))
    ax.quiver(points[:, 0], points[:, 1], points[:, 2], vectors[:, 0], vectors[:, 1], vectors[:, 2], length=5,
              normalize=True, linewidths=3)
    ax.set_xlabel('X', fontsize=20, rotation=150)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('Z', fontsize=30, rotation=60)
    plt.show()

    # SAVE RT VECTORS AND TIMES
    np.save(f'{save_folder}/tvecs', np.array(tvecs))
    np.save(f'{save_folder}/rvecs', np.array(rvecs))
    np.save(f'{save_folder}/times', np.array(times))

    #np.save(f'{save_folder}/points', points)
    #np.save(f'{save_folder}/vectors', vectors)

    TRACKER.stop_tracking()
    TRACKER.close()


def main():
    project_path = Path(__file__).parent.resolve()

    # folder will be structured as follows:
    # assets/type/folder/images
    type = 'tests'  # random / phantom / EM_tracker_calib / tests
    folder = 'EM_half_square'

    save_folder = f'{project_path}/assets/{type}/{folder}'

    # Capture_all:
    # True will capture every single frame (with frame rate specified)
    # False: you will have to press the key 'c' to capture a frame
    CAPTURE_ALL = False
    # how often to record frames- only applies if CAPTURE_ALL is true
    FRAME_RATE = 10

    # max frames that will be recorded
    MAX_FRAMES = 3000

    # where intrinsics and distortion are located
    calibration_path = f'{project_path}/calibration/endoscope_calibration'
    record_data_EM(save_folder, calibration_path, capture_all=CAPTURE_ALL, frame_rate=FRAME_RATE, max_frames=MAX_FRAMES)

if __name__ == "__main__":
    main()