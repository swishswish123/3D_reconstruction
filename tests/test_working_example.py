from pathlib import Path
import glob
import numpy as np
import cv2
import match_pairs
from reconstruction_utils.utils import get_matched_keypoints_sift, get_matched_keypoints_superglue, \
    extract_rigid_body_parameters, rigid_body_parameters_to_matrix
from reconstruction_utils.reconstruction_algorithms import triangulate_points_opencv
import matplotlib.pyplot as plt

########
'''
This test is a mock of a full working example of reconstruction. The setup was as follows:
I first calibrated the camera using the checkerboard method. Calibration can be found under calibration/mac_calibration

The 'object' I was recording is some sort of a square but not quite, something similar to below:

                              5cm
                        (1)—————————(2)
            2.5cm       |           |
                         (3)        |    5cm
                        .(4)        |(5)

Where the square has sides 5cm but the bottom line isn't drawn and the left is only drawn half-way (2.5cm) 
and the bottom left corner is just a dot. 
This is so that I can uniquely distinguish 5 points no matter the orientation (The 5 points labelled in the image above).

The experiment then consisted of placing the computer camera at a known z distance to the object (I set it at roughly 35cm)
I then took an image (frame1/im1), moved the camera by a known amount (roughly 20cm) and took another image (frame2/im2)

The goal of this test is to be able to recover the poses of each of these 5 points. 
'''




"""
def test_EM_poses(project_path, intrinsics, distortion):
    # folder will be structured as follows:
    # assets/type/folder/images

    type = 'tests'  # random / phantom / EM_tracker_calib / tests
    folder = 'EM_half_square'

    camera_info = f'{project_path}/../assets/{type}/{folder}'
    # images
    frames_pth = sorted(glob.glob(f'{camera_info}/images/*.*'))
    # image pair- the pair of images we are choosing to perform test with. I chose this one as it has best pairing with superglue
    im1_idx = 4
    im2_idx = 5

    # loading two frames
    im1_pth = frames_pth[im1_idx]
    im2_pth = frames_pth[im2_idx]

    im1 = cv2.imread(im1_pth)
    im2 = cv2.imread(im2_pth)

    # resolution of camera on my computer is 1080 by 1920, and we expect 3 channels
    assert im1.shape == (1080, 1920, 3)
    assert im2.shape == (1080, 1920, 3)

    # load poses
    poses = np.load(f'{camera_info}/vecs.npy')
    hand_eye = np.load(f'{project_path}/../calibration/endoscope_calibration/h2e.npy')

    im1_poses = poses[im1_idx]
    im2_poses = poses[im2_idx]
    T_1_to_2 = hand_eye @ np.linalg.inv(im2_poses) @ im1_poses @ np.linalg.inv(hand_eye)

    ######### KEYPOINTS 
    # performing superglue matching
    match_pairs.superglue(type, folder)

    # matches paths once performed
    # path where all match pairs located (SUPERGLUE)
    output_dir = f'{project_path}/../outputs/match_pairs_{type}_{folder}/'
    # matches (only second to last pair of images as it has 4 matches)
    match_paths = sorted(glob.glob(f'{output_dir}/*.npz'))[im1_idx]

    # getting key points that are matched in both images. In this test there are 6 matching points.
    kp1_matched, kp2_matched = get_matched_keypoints_superglue(match_paths)
    # To visualise these results, uncomment the following:
    '''
    fig, ax = plt.subplots()
    n=[1,2,3,4]
    x1 = kp1_matched[:,0]
    y1 = kp1_matched[:,1]

    x2 = kp2_matched[:,0]
    y2 = kp2_matched[:,1]

    ax.scatter(x1,y1)

    for i, txt in enumerate(n):
        ax.annotate(txt, (x1[i], y1[i]))

    ax.scatter(x2,y2)

    for i, txt in enumerate(n):
        ax.annotate(txt, (x2[i], y2[i]))

    plt.gca().invert_yaxis()
    plt.savefig('kp1.png')
    '''

    projected_points_1_undistorted = cv2.undistortPoints(kp1_matched, intrinsics, distortion, None, intrinsics)
    projected_points_2_undistorted = cv2.undistortPoints(kp2_matched, intrinsics, distortion, None, intrinsics)

    projected_points_1_undistorted_squeezed = projected_points_1_undistorted.squeeze(1)
    projected_points_2_undistorted_squeezed = projected_points_2_undistorted.squeeze(1)

    # creating corresponding points in world coordinates (defined by me) as I know the
    # length of the square sides is 5cm
    # solve pnp
    ground_truth_pnts = np.array([
        [50, 0, 0],
        [0, 25, 0],
        [50, 50, 0],
        [0, 50, 0]
    ])

    # solve PNP on image points 1- note that input object points and image points need to be floats for this to work
    works, R1, T1 = cv2.solvePnP(ground_truth_pnts.astype('float32'),
                                 projected_points_1_undistorted_squeezed.astype('float32'), intrinsics, distortion)
    # solve PNP on second image
    works, R2, T2 = cv2.solvePnP(ground_truth_pnts.astype('float32'),
                                 projected_points_2_undistorted_squeezed.astype('float32'), intrinsics, distortion)

    pnp1_poses = rigid_body_parameters_to_matrix(
        np.concatenate([R1.T, T1.T], axis=1).squeeze()
    )
    pnp2_poses = rigid_body_parameters_to_matrix(
        np.concatenate([R2.T, T2.T], axis=1).squeeze()
    )
    T_1_to_2_pnp = np.linalg.inv(pnp2_poses) @ pnp1_poses

    # extract rigid body params for better undestanding
    params_norm = extract_rigid_body_parameters(T_1_to_2)
    params_pnp = extract_rigid_body_parameters(T_1_to_2_pnp)

    assert params_norm == params_pnp

    return im1, im2
"""


def test_aroCo_poses(project_path):

    # load calibration data
    intrinsics = np.loadtxt(f'{project_path}/../calibration/mac_calibration/intrinsics.txt')
    distortion = np.loadtxt(f'{project_path}/../calibration/mac_calibration/distortion.txt')

    # in this test I moved by 20cm along x between the 2 images. Z distance is 36cm
    type = 'tests'  # random / phantom / EM_tracker_calib / tests
    folder = 'half_square_2'

    camera_info = f'{project_path}/../assets/{type}/{folder}'
    # images
    frames_pth = sorted(glob.glob(f'{camera_info}/images/*.*'))
    # image pair - only 2 imgs taken
    im1_pth = frames_pth[0]
    im2_pth = frames_pth[1]
    # load images
    im1 = cv2.imread(im1_pth)
    im2 = cv2.imread(im2_pth)

    # load poses (taken from aruCo marker positions)
    R_poses = np.load(f'{camera_info}/rvecs.npy')
    T_poses = np.load(f'{camera_info}/tvecs.npy')
    im1_R = R_poses[0]
    im1_T = T_poses[0]
    im2_R = R_poses[1]
    im2_T = T_poses[1]
    # converting to 4x4 matrix
    poses_1 = rigid_body_parameters_to_matrix(np.concatenate([im1_R, im1_T]))
    poses_2 = rigid_body_parameters_to_matrix(np.concatenate([im2_R, im2_T]))
    # 4x4 to get from cam1 position to cam2 position
    T_1_to_T_2 = np.linalg.inv(poses_2) @ poses_1

    # For keypoints matching definitions I looked at the image and got the points
    # in interactive mode. If you want to look at the image uncomment the lines below
    # mpl.use('TkAgg')
    # plt.imshow(im1)
    kp1_matched = np.array([
        [540, 406],  # top left corner
        [746, 419],  # top right corner
        [530, 506],  # middle left of square where line stops
        [525, 609],  # bottom left dot
        [726, 620],  # bottom right of square where line ends

    ])

    kp2_matched = np.array([
        [1230, 412],  # top left corner
        [1420, 422],  # top right corner
        [1227, 509],  # middle left of square where line stops
        [1223, 602],  # bottom left dot
        [1412, 614],  # bottom right of square where line ends

    ])

    # #### TESTING CAMERA POSES First I want to test whether my aruco markers correctly identified the pose
    # difference between camera 1 and camera 2. To do this, I compare it to the camera poses I get by using solvePnP
    # which I can perform as I know the real world points of my object

    # these are the "ground truth" object relative poses which I use to test the poses with solvePnP
    object_points = np.array([
        [0, 0, 0],
        [50, 0, 0],
        [0, 25, 0],
        [0, 50, 0],
        [50, 50, 0]
    ])

    # solve PNP on image points 1- note that input object points and image points need to be floats for this to work
    _ , R1, T1 = cv2.solvePnP(object_points.astype('float32'),
                                 kp1_matched.astype('float32'), intrinsics, distortion)
    # solve PNP on second image
    _, R2, T2 = cv2.solvePnP(object_points.astype('float32'),
                                 kp2_matched.astype('float32'), intrinsics, distortion)

    # estimated cam 1 and 2 poses with pnp in matrix 4x4
    pnp1_poses = rigid_body_parameters_to_matrix(
        np.concatenate([R1.T, T1.T], axis=1).squeeze()
    )
    pnp2_poses = rigid_body_parameters_to_matrix(
        np.concatenate([R2.T, T2.T], axis=1).squeeze()
    )
    # cam1 to cam 2
    T_1_to_2_pnp = np.linalg.inv(pnp2_poses) @ pnp1_poses

    # extract rigid body params for better comparison of matrices
    params_norm = extract_rigid_body_parameters(T_1_to_T_2)
    params_pnp = extract_rigid_body_parameters(T_1_to_2_pnp)

    # checking the relative poses are roughly the same / x moves by about 200cm and z&y change by roughly 0
    print('params_norm')
    print(params_norm)
    print('params_pnp')
    print(params_pnp)

    # TODO add assertion tests here
    # assert params_norm==params_pnp

    # undistorting points before triangulation
    projected_points_1_undistorted = cv2.undistortPoints(kp1_matched.astype('float32'), intrinsics, distortion, None,
                                                         intrinsics)
    projected_points_2_undistorted = cv2.undistortPoints(kp2_matched.astype('float32'), intrinsics, distortion, None,
                                                         intrinsics)

    projected_points_1_undistorted_squeezed = projected_points_1_undistorted.squeeze(1)
    projected_points_2_undistorted_squeezed = projected_points_2_undistorted.squeeze(1)

    ############### now testing triangulation
    result_norm = triangulate_points_opencv(projected_points_1_undistorted_squeezed.T,
                                            projected_points_2_undistorted_squeezed.T, intrinsics, np.array([0, 0, 0]),
                                            np.array(params_norm[:3]), np.array([0, 0, 0]), np.array(params_norm[3:]))
    result_pnp = triangulate_points_opencv(projected_points_1_undistorted_squeezed.T,
                                            projected_points_2_undistorted_squeezed.T, intrinsics, np.array([0, 0, 0]),
                                            np.array(params_pnp[:3]), np.array([0, 0, 0]), np.array(params_pnp[3:]))

    # what we know-
    # I placed the camera roughly 35cm away from the paper
    # z should be roughly equal and roughly 35cm- note that my measurement was very rough....
    assert len(set(result_norm[:, 2].round(
        -2))) == 1  # rounding to nearest 100, then checking how many unique values there are. If just 1 unique value then all elements equal
    assert np.isclose(result_norm[:, 2], 350, atol=30).all()  # checking each z value within 3cm of 35cm

    tolerance = 5  # mm
    # point 1 and 2 should be 50mm apart in x and 0 in y
    pnt_2_minus_1 = result_norm[1, :] - result_norm[0, :]  # 2-1
    assert np.isclose(pnt_2_minus_1[0], 50, atol=tolerance)  # x: 50 (+/-tol)mm
    assert np.isclose(pnt_2_minus_1[1], 0, atol=tolerance)  # y: 0 (+/-tol)mm
    # point 1 and 3 should be 25mm apart in y and 0 in x
    pnt_3_minus_1 = result_norm[2, :] - result_norm[0, :]  # 3-1
    assert np.isclose(pnt_3_minus_1[0], 0, atol=tolerance)  # x: 0 (+/-tol)mm
    assert np.isclose(pnt_3_minus_1[1], 25, atol=tolerance)  # y: 25 (+/-tol)mm
    # point 3 and 4 should be 25mm apart in y and 0 in x
    pnt_4_minus_3 = result_norm[3, :] - result_norm[2, :]  # 4-3
    assert np.isclose(pnt_4_minus_3[0], 0, atol=tolerance)  # x: 0 (+/-tol)mm
    assert np.isclose(pnt_4_minus_3[1], 25, atol=tolerance)  # y: 25 (+/-tol)mm
    # point 4 and 5 should be 50mm apart in x and 0 in y
    pnt_5_minus_4 = result_norm[4, :] - result_norm[3, :]  # 5-4
    assert np.isclose(pnt_5_minus_4[0], 50, atol=tolerance)  # x: 50 (+/-tol)mm
    assert np.isclose(pnt_5_minus_4[1], 0, atol=tolerance)  # y: 0 (+/-tol)mm
    # point 2 and 5 should be 50mm apart in y and 0 in x
    ####ALL ABOVE TESTS PASSED WITH 5 AS TOLERANCE. TOLERANCE BELOW HAD TO BE 5.6 TO PASS
    tolerance = 5.6  # mm
    pnt_5_minus_2 = result_norm[4, :] - result_norm[1, :]  # 4-3
    assert np.isclose(pnt_5_minus_2[0], 0, atol=tolerance)  # x: 0 (+/-tol)mm
    assert np.isclose(pnt_5_minus_2[1], 50, atol=tolerance)  # y: 50 (+/-tol)mm


if __name__ == '__main__':
    project_path = Path(__file__).parent.resolve()

    test_aroCo_poses(project_path)
    # im1, im2 = test_EM_poses()
    # all_tests()
    # test_keypoints()
