# this file will perform a sparse 3D reconstruction. The output will be 3D points and colors (RGB)
# you only need to adjust any variables under 'utils'

import os
import glob
from pathlib import Path
import cv2
import matplotlib as mpl
import match_pairs
import numpy as np


from reconstruction_utils.reconstruction_algorithms import triangulate_points_opencv, get_xyz, get_xyz_method_prince

from reconstruction_utils.utils import get_matched_keypoints_superglue, get_matched_keypoints_sift, extrinsic_matrix_to_vecs, extrinsic_vecs_to_matrix, estimate_camera_poses, manually_match_features, multiply_points_by_transform

from reconstruction_utils.bundle_adjustment import perform_bundle_adjustment


def reconstruct_pairs(reconstruction_method, kp1_matched, kp2_matched, intrinsics, T1_to_T2):
    """
    Function to obtain 3D triangulated points from an image pair

    Args:
        reconstruction_method: method used for reconstruction- opencv, prince, online estimate_pose
        kp1_matched: keypoints in image 1 which are matched to image 2
        kp2_matched: keypoints in image 2 which are matched to image 1
        intrinsics (3x3): numpy array containing intrinsics matrix of camera
        T1_to_T2 (4x4): transform matrix between two image pairs camera poses

    Returns:
        D3_points (#TODO CHECK DIMS):
        colour_mask (): array of all the indeces that were correctly triangulated. This
        will be used to filter out the colour points whose points weren't used
        #TODO check this is working
    """

    # rotation and translation vec of first frame will be set to zero, R and t of second
    # camera position will be set to whatever the transform between the two cameras is
    # - calculated below (T_1_to_T_2)
    rvec_1 = np.zeros(3)
    tvec_1 = np.zeros(3)
    T0 = extrinsic_vecs_to_matrix(rvec_1, tvec_1)
    colour_mask = None
    if reconstruction_method == 'opencv':
        '''
        triangulate with opencv function
        '''
        # rotation and translation vectors between two frames in euler angles
        rvec_2, tvec_2 = extrinsic_matrix_to_vecs(T1_to_T2)
        D3_points = triangulate_points_opencv(kp1_matched, kp2_matched, intrinsics, rvec_1, rvec_2, tvec_1, tvec_2)

    elif reconstruction_method == 'prince':
        '''
        triangulating with prince method 
        '''
        D3_points, colour_mask = get_xyz_method_prince(intrinsics, kp1_matched, T0, kp2_matched, T1_to_T2)

    elif reconstruction_method == 'online':

        D3_points = get_xyz(kp1_matched, intrinsics, T0[:3, :3], T0[:3, 3:], kp2_matched, intrinsics,
                            T1_to_T2[:3, :3], T1_to_T2[:3, 3:])

    return D3_points, colour_mask


def get_image_poses(tracking_method ,idx=None, frame_rate=None, poses=None, hand_eye=None,rvecs=None, tvecs=None, kp1_matched=None, kp2_matched=None, intrinsics=None):
    """
    Computes the relative transformation matrix between two camera poses using different tracking methods.

    Parameters:
    -----------
    tracking_method : str
        The method used for tracking the camera poses. It can be 'EM' for extrinsic matrix,
         'aruCo' for ArUco markers, or any other value for feature-based tracking
         (pose estimation).
    idx : int or None, optional
        The index of the first image pose in the poses array. This is only required for
        EM and ArUco methods, where the poses array contains multiple poses.
    frame_rate : int or None, optional
        The number of frames between the two camera poses. This is only required for EM and ArUco methods.
    poses : numpy.ndarray or None, optional
        An array containing the camera poses. This is only required for EM and ArUco methods.
    hand_eye : numpy.ndarray or None, optional
        A 4x4 homogeneous transformation matrix representing the hand-eye calibration. This is only required for EM method.
    rvecs : numpy.ndarray or None, optional
        A 3x1 array containing the rotation vector of the camera pose. This is only required for ArUco method.
    tvecs : numpy.ndarray or None, optional
        A 3x1 array containing the translation vector of the camera pose. This is only required for ArUco method.
    kp1_matched : numpy.ndarray or None, optional
        An array containing the keypoints in the first image. This is only required for feature-based tracking.
    kp2_matched : numpy.ndarray or None, optional
        An array containing the keypoints in the second image. This is only required for feature-based tracking.
    intrinsics : numpy.ndarray or None, optional
        A 3x3 array containing the camera intrinsic matrix. This is only required for feature-based tracking.

    Returns:
    --------
    T1_to_T2 : numpy.ndarray
        A 4x4 homogeneous transformation matrix representing the relative transformation between the two camera poses.
    rvec_2 : numpy.ndarray or None
        A 3x1 array containing the rotation vector of the second camera pose. This is only returned when the tracking method is feature-based.
    tvec_2 : numpy.ndarray or None
        A 3x1 array containing the translation vector of the second camera pose. This is only returned when the tracking method is feature-based.
    """

    if tracking_method == 'EM':
        # selecting poses information of current img pairs
        ################## 2))))))))))))))
        im1_poses = poses[idx]
        #im1_poses = poses[0]
        im2_poses = poses[idx + frame_rate]
        # getting relative transform between two camera poses
        # relative position between the two is going from the first image to the origin,
        # then from origin to the second image
        T1_to_T2 = hand_eye @ np.linalg.inv(im2_poses) @ im1_poses @ np.linalg.inv(hand_eye)

    elif tracking_method == 'aruCo':
        # selecting poses of aruco of current frame pair and converting to 4x4 matrix
        ################## 3)))))))))))))))))))
        im1_mat = extrinsic_vecs_to_matrix(rvecs[idx], tvecs[idx]) #(4x4)
        #im1_mat = extrinsic_vecs_to_matrix(rvecs_1, tvecs_1)  # (4x4)
        im2_mat = extrinsic_vecs_to_matrix(rvecs[idx + frame_rate], tvecs[idx + frame_rate])  # (4x4)
        # relative transform between two camera poses
        T1_to_T2 = im2_mat @ np.linalg.inv(im1_mat)  # (4x4)
    else:
        rvec_1 = np.zeros(3) #(3,)
        tvec_1 = np.zeros(3) #(3,)

        # estimating camera poses and converting rotation to vector
        R2, tvec_2 = estimate_camera_poses(kp1_matched, kp2_matched, intrinsics) # R2: (3x3), tvec_2: (3x1)
        rvec_2,_ = cv2.Rodrigues(R2) # (3x1)
        # getting 4x4 camera poses
        im1_poses = extrinsic_vecs_to_matrix(rvec_1, tvec_1) #(4x4)
        im2_poses = extrinsic_vecs_to_matrix(rvec_2, tvec_2) #(4x4)
        #TODO change this to correct transform back to original poses
        T1_to_T2 = im2_poses @ np.linalg.inv(im1_poses)  # (4x4)
        return T1_to_T2, rvec_2, tvec_2

    return T1_to_T2


def sparse_reconstruction_from_video(data_path,calibration_path,save_path,
                                     frame_rate=1,
                                     tracking_method='aruCo',
                                     matching_method='superglue',
                                     reconstruction_method='opencv',
                                     bundle_adjust=False

                                     ):
    """ Function performs 3D sparse reconstruction from set of images with overlapping bits

    Provided a set of images and camera calibration data, this function will save a 3D point cloud
    and a 3D colour cloud to the save_path specified.
    The sparse points are points that could be matched between pairs of images and subsequently triangulated.

    Args:
        data_path (str): path where images folder and pose information are located
        calibration_path (str): path whrere intrinsics.txt and distortion.txt data is located
        save_path (str): path where output point cloud will be stored
        frame_rate (int): rate at which to select frames for matching and reconstruction
        tracking_method (str): method used for tracking. This can be either 'EM', 'aruCo'
        matching_method (str): method used for matching between images. Options are 'superglue', 'sift' or 'manual'. The manual method
                                will bring up a subplot where you will be able to select points that are matching between the
                                images.
        reconstruction_method (str): method to be used for reconstruction- opencv is the opencv.triangulate() and will be used as a default.
        bundle_adjust (bool): whether or not to perform bundle adjustment
    Returns:
        this function will save two point clouds to the save folder specified:
        - points.npy (NX3): points numpy array- 3 dimensions representing X,Y,Z
        - colors.npy (NX3): points numpy array- 3 dimensions representing R,G,B
    """

    mpl.use('TkAgg')

    ########################## LOADING ALL ###################################
    project_path = Path(__file__).parent.resolve()
    type = data_path.split('/')[-2] # folder above images
    folder = data_path.split('/')[-1] # 2 folders above images

    # images
    frames_pth = sorted(glob.glob(f'{data_path}/images/*.*'))
    # calibration data
    intrinsics = np.loadtxt(f'{calibration_path}/intrinsics.txt')
    distortion = np.loadtxt(f'{calibration_path}/distortion.txt')

    # if we are tracking we need to load camera pose info
    if tracking_method == 'EM':
        # camera poses
        poses = np.load(f'{data_path}/vecs.npy')
        # calibration data
        hand_eye = np.load(f'{calibration_path}/h2e.npy')

    elif tracking_method == 'aruCo':
        rvecs = np.load(f'{data_path}/rvecs.npy')
        tvecs = np.load(f'{data_path}/tvecs.npy')

        if len(rvecs) == 0 or len(tvecs) == 0:
            ValueError('rvecs or tvecs empty- no tracking data')

    # creating path where to save reconstructions
    reconstruction_output = f'{save_path}'
    if not os.path.isdir(reconstruction_output):
        os.makedirs(reconstruction_output)

    # ####################### PERFORMING SUPERGLUE MATCHING ########################################
    # if superglue is selected, superglue matching is performed before running and output is saved under outputs
    if matching_method == 'superglue':
        # performing superglue matching
        match_pairs.superglue(data_path, frame_rate=frame_rate)

        # matches paths once performed
        # path where all match pairs located (SUPERGLUE)
        output_dir = f'{project_path}/assets/superglue_matches/match_pairs_{type}_{folder}'
        # matches
        # match_paths = glob.glob(f'{output_dir}/*.npz')

    # ##################### PERFORM 3D RECONSTRUCTION FOR EACH FRAME######################

    D3_points_all = []
    D3_colors_all = []

    # params for bundle adjustment
    # rotation vector (3 elements)
    # translation vector (3 elements)
    # focal length (1 element)
    # principal point x-coordinate (1 element)
    # principal point y-coordinate (1 element)
    # [rx, ry, rz, tx, ty, tz, fx, fy, cx]
    camera_params = []
    start_idx = 0 # index at which to start next count of 3d points
    camera_indeces = []
    point_indeces = []
    points2d = [] # all 2d points

    # this will be the 4x4 matrix to go from original position to current frame
    TN_to_T1 = np.eye(4)
    for idx in np.arange(0, len(frames_pth) - 1, frame_rate):
    #for idx in [0]:
        if idx == 0:
            camera_params.append([0,0,0,0,0,0,intrinsics[0,0], distortion[0], distortion[1]])

        if idx % 10 == 0:
            print(f'image {idx}')

        # frames path of two matched pairs
        ##################### 1)))))))))))))))))
        im1_path = frames_pth[idx]
        #im1_path = frames_pth[0]
        im2_path = frames_pth[idx + frame_rate]

        # image numbers (eg. 00000001)- excluding extension
        im1 = im1_path.split('/')[-1][:-4]
        im2 = im2_path.split('/')[-1][:-4]

        # loading image pairs we're reconstructing
        img1_original = cv2.imread(im1_path)
        img2_original = cv2.imread(im2_path)

        img1_original = cv2.cvtColor(img1_original, cv2.COLOR_BGR2RGB)
        img2_original = cv2.cvtColor(img2_original, cv2.COLOR_BGR2RGB)

        ############################### MATCHING ##########################
        # obtaining or loading keypoints between images
        # kpn_matched will be (N*2)
        if matching_method == 'superglue':
            kp1_matched, kp2_matched = get_matched_keypoints_superglue(f'{output_dir}/{im1}_{im2}_matches.npz')
        elif matching_method == 'sift':
            kp1_matched, kp2_matched = get_matched_keypoints_sift(img1_original, img2_original)
        elif matching_method == 'manual':
            kp1_matched, kp2_matched = manually_match_features(img1_original, img2_original)
            print(kp1_matched, kp2_matched)

        # if no matches found skip to next frame pair
        if np.size(kp1_matched) == 0:
            print('no matches')
            continue

        ######################## undistort keypoints ########################################################
        kp1_matched = cv2.undistortPoints(kp1_matched.astype('float32'), intrinsics, distortion, None,
                                          intrinsics)
        kp2_matched = cv2.undistortPoints(kp2_matched.astype('float32'), intrinsics, distortion, None,
                                          intrinsics)

        kp1_matched = kp1_matched.squeeze(1).T  # (2XN)
        kp2_matched = kp2_matched.squeeze(1).T  # (2XN)

        #points2d.append(np.stack((kp1_matched.T, kp2_matched.T), axis=1))
        # adding keypoints to list of kpts
        points2d += np.ndarray.tolist(kp1_matched.T)
        # camera used to obtain all these points is current idx
        num_keypoints = kp1_matched.shape[1]
        camera_indeces += np.ndarray.tolist( idx * np.ones(num_keypoints) )

        # adding keypoints to list of kpts
        points2d += np.ndarray.tolist(kp2_matched.T)
        # camera used to obtain all these points is current idx
        camera_indeces += np.ndarray.tolist( (idx+1) * np.ones(kp2_matched.shape[1]) )

        # point indeces will be- where we left off until the length of the kpts (this is repeated twice as we have the 2 kpts representing the same 3d triangulation
        point_indeces += np.ndarray.tolist( np.arange(start_idx, start_idx+num_keypoints, 1)) *2
        start_idx += num_keypoints

        ####################### GETTING COLOR OF POINTS ###############################################
        # selecting colors of keypoints which will be triangulated
        input_undistorted_points = np.concatenate([kp1_matched.T, kp2_matched.T], axis=1)
        input_undistorted_points = input_undistorted_points.astype(int)  # converting to integer
        D3_colors = img1_original[
            input_undistorted_points[:, 1], input_undistorted_points[:, 0]]  # (Nx3)- where N is same as kp_matched N

        if not idx == 0:
            TN_to_T1 =  TN_to_TN_plus_1 @ TN_to_T1
            # first camera as reference

        # ########################### IMAGE POSES ################################
        # obtaining image posees between current frame (N) and the next (N+1)
        if tracking_method == 'EM':
            TN_to_TN_plus_1 = get_image_poses(tracking_method, idx=idx, frame_rate=frame_rate, poses=poses, hand_eye=hand_eye)
        elif tracking_method =='aruCo':
            TN_to_TN_plus_1 = get_image_poses(tracking_method, idx=idx, frame_rate=frame_rate, rvecs=rvecs, tvecs=tvecs, kp1_matched=None, kp2_matched=None, intrinsics=None)
        else: # estimate_pose
            TN_to_TN_plus_1, R_est, T_est = get_image_poses(tracking_method,  kp1_matched=kp1_matched, kp2_matched=kp2_matched, intrinsics=intrinsics)
            #poses_all.append(np.hstack([R_est.T, T_est.T]))
            #poses_all.append(TN_to_TN_plus_1)

            # ## camera parameters for later using in bundle adjustment

            #camera_index += 1
        R_est, T_est = extrinsic_matrix_to_vecs(TN_to_TN_plus_1)

        camera_params.append(
            [R_est[0, 0], R_est[1, 0], R_est[2, 0],
             T_est[0, 0], T_est[1, 0], T_est[2, 0],
             intrinsics[0, 0],
             distortion[0], distortion[1]])
        # ######################## TRIANGULATION- GETTING 3D POINTS ##################################
        D3_points, color_mask = reconstruct_pairs(reconstruction_method, kp1_matched, kp2_matched, intrinsics, TN_to_TN_plus_1)
        # camera indeces-

        bring_back = True
        if bring_back:
            # TODO- ADD CHECK IF THERE'S NO 3D POINTS
            D3_points = multiply_points_by_transform(D3_points, np.linalg.inv(TN_to_T1))

        #points_all.append(D3_points)
        D3_points_all += np.ndarray.tolist(D3_points)

        if color_mask:
            D3_colors_all += np.ndarray.tolist(D3_colors[color_mask])
        else:
            D3_colors_all += np.ndarray.tolist(D3_colors)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # convert points to array for all point cloud
    point_indeces = np.asarray(point_indeces, dtype='int')
    camera_indeces = np.asarray(camera_indeces, dtype='int')
    points2d = np.asarray(points2d, dtype='float64')
    # reconstructed pointcloud- coords then colors
    all_points = np.asarray(D3_points_all, dtype='float64')
    all_colors = np.asarray(D3_colors_all, dtype='float')
    camera_params = np.asarray(camera_params, dtype='float64')

    # output saved as (NX3) points numpy array- 3 dimensions representing X,Y,Z
    np.save(f'{reconstruction_output}/points.npy', all_points)
    # output saved as (NX3) points numpy array- 3 dimensions representing R,G,B
    np.save(f'{reconstruction_output}/colors.npy', all_colors)

    # performing bundle adjustment
    if bundle_adjust:
        poses_opt, points3D_opt = perform_bundle_adjustment(camera_params, camera_indeces, point_indeces, points2d, all_points)
        print(points3D_opt)
        # improved 3d points
        np.save(f'{reconstruction_output}/points_opt.npy', points3D_opt)
        print('done')


def main():
    project_path = Path(__file__).parent.resolve()
    
    ########################## PARAMS ###################################
    # method of performing 3D reconstruction
    method = 'opencv'  # opencv/sksurgery/online/prince
    # parent folder of data type
    type = 'aruco'  # random / phantom / EM_tracker_calib /tests
    # folder right on top of images
    folder = 'shelves_video'
    # folder where all camera info stored- images and poses
    data_path = f'{project_path}/assets/data/{type}/{folder}'
    print(data_path)
    # folder where reconstruction pointclouds are stored
    save_path = f'{project_path}/reconstructions/{method}/{type}/{folder}'

    # RANDOM, UNDISTORTED: arrow / brain  / checkerboard_test_calibrated / gloves /
    # RANDOM UNDISTORTED MAC CAM: mac_camera /
    # RANDOM, Distorted: books / points / spinal_section / spinal_section_pink
    # EM_TRACKING_CALIB testing_points /testing_lines
    # PHANTOM: surface / right_in / phantom_surface_2 / both_mid
    # tests: half_square_2 / EM_half_square
    # aruco: puzzle_box /shelves (70 z, 70 x, 20y) /shelves_2 (25x, 0y, 95z)

    # determines space between images we pick
    frame_rate = 1
    # tracking type we're using
    TRACKING = None  # EM / aruCo / False

    # change to the correct folder where intrinsics and distortion located
    calibration_path = f'{project_path}/calibration/mac_calibration/'

    # method of getting keypoints and matching between frames
    # sift- sift for feature matching
    # superglue- uses superglue for feature matching
    # manual- you will manually label corresponding points between frames
    matching_method = 'superglue'  # sift / superglue / manual

    sparse_reconstruction_from_video(data_path,calibration_path,save_path,
                                     frame_rate=frame_rate,
                                     tracking_method = TRACKING,
                                     matching_method=matching_method,
                                     reconstruction_method=method,
                                     bundle_adjust=True)

if __name__=='__main__':
    main()
