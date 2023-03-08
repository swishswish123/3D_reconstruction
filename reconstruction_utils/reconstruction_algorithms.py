

import numpy as np
from pathlib import Path
import os
import glob
import matplotlib.pyplot as plt
import cv2
from matplotlib import lines
import matplotlib.cm as cm
from models.utils import plot_keypoints, plot_matches,process_resize
from scipy.spatial.transform import Rotation as spr
import sksurgerycore.transforms.matrix as stm
import match_pairs

from .utils import *

'''
def reconstruction_gpt(points1, points2, K, R1, t1, R2, t2):
    # Define world coordinate system
    world_origin = np.array([0, 0, 0])
    world_x_axis = np.array([1, 0, 0])
    world_y_axis = np.array([0, 1, 0])
    world_z_axis = np.array([0, 0, 1])

    # Define camera 1 pose relative to world coordinate system
    T1 = np.eye(4)
    T1[:3, :3] = R1
    T1[:3, 3] = t1
    T1_w = np.eye(4)
    T1_w[:3, :3] = world_x_axis, world_y_axis, world_z_axis
    T1_w[:3, 3] = world_origin
    T1_cw = np.linalg.inv(T1_w) @ T1

    # Define camera 2 pose relative to world coordinate system
    T2 = np.eye(4)
    T2[:3, :3] = R2
    T2[:3, 3] = t2
    T2_w = np.eye(4)
    T2_w[:3, :3] = world_x_axis, world_y_axis, world_z_axis
    T2_w[:3, 3] = world_origin
    T2_cw = np.linalg.inv(T2_w) @ T2

    # Triangulate points
    points1_norm = cv2.undistortPoints(points1, K, None)
    points2_norm = cv2.undistortPoints(points2, K, None)
    points1_norm_hom = np.hstack((points1_norm.reshape(-1, 2), np.ones((points1_norm.shape[0], 1))))
    points2_norm_hom = np.hstack((points2_norm.reshape(-1, 2), np.ones((points2_norm.shape[0], 1))))
    
    points1_norm = points1_norm_hom[:, :2] / points1_norm_hom[:, 2:]
    points2_norm = points2_norm_hom[:, :2] / points2_norm_hom[:, 2:]
    points_3d_hom = cv2.triangulatePoints(T1_cw[:3], T2_cw[:3], points1_norm.T, points2_norm.T)
    points_3d_hom /= points_3d_hom[3, :]
    points_3d = points_3d_hom[:3, :].T

    return points_3d
'''


'''
def reconstruction_gpt(kp1_matched, kp2_matched, intrinsics, T_1_to_2):
    # Convert matched keypoints to homogeneous coordinates
    kp1_homogeneous = cv2.convertPointsToHomogeneous(kp1_matched).squeeze()
    kp2_homogeneous = cv2.convertPointsToHomogeneous(kp2_matched).squeeze()

    # Calculate projection matrices for the two views
    P1 = intrinsics @ np.hstack((np.identity(3), np.zeros((3, 1))))
    P2 = intrinsics @ T_1_to_2[:3, :]

    # Triangulate 3D points from the corresponding 2D points
    output_points_homogeneous = cv2.triangulatePoints(P1, P2, kp1_homogeneous, kp2_homogeneous)

    # Convert the points back to Cartesian coordinates
    output_points_cartesian = output_points_homogeneous[:3, :] / output_points_homogeneous[3, :]

    return output_points_cartesian.T
'''



'''
def triangulate_points_opencv_2(kp1_matched, kp2_matched, intrinsics, T_1_to_2):
    
    P1 = intrinsics @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.double)
    P2 = intrinsics @ T_1_to_2[:3,:]

    output_points = cv2.triangulatePoints(P1, P2, kp1_matched.T, kp2_matched.T)

    return (output_points[:3] / output_points[3] ).T
'''

# Initialize consts to be used in linear_LS_triangulation()


def linear_LS_triangulation(u1, P1, u2, P2):
    """
    https://github.com/Eliasvan/Multiple-Quadrotor-SLAM/blob/master/Work/python_libs/triangulation.py
    Linear Least Squares based triangulation.
    Relative speed: 0.1
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector will be True for all points.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.zeros((3, len(u1)))
    
    # Initialize C matrices
    linear_LS_triangulation_C = -np.eye(2, 3)
    C1 = np.array(linear_LS_triangulation_C)
    C2 = np.array(linear_LS_triangulation_C)
    
    for i in range(len(u1)):
        # Derivation of matrices A and b:
        # for each camera following equations hold in case of perfect point matches:
        #     u.x * (P[2,:] * x)     =     P[0,:] * x
        #     u.y * (P[2,:] * x)     =     P[1,:] * x
        # and imposing the constraint:
        #     x = [x.x, x.y, x.z, 1]^T
        # yields:
        #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
        #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
        # and since we have to do this for 2 cameras, and since we imposed the constraint,
        # we have to solve 4 equations in 3 unknowns (in LS sense).

        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[i, :]
        C2[:, 2] = u2[i, :]
        
        # Build A matrix:
        # [
        #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
        #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
        #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
        #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
        # ]
        A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2
        
        # Build b vector:
        # [
        #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
        #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
        #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
        #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
        # ]
        b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
        b *= -1
        
        # Solve for x vector
        cv2.solve(A, b, x[:, i:i+1], cv2.DECOMP_SVD)
    
    return x.T.astype(dtype='float'), np.ones(len(u1), dtype=bool)


def normalize_keypoints(norm_kpts, camera_matrix, dist_coeffs):
    # Get camera parameters
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    #dist_coeffs = camera_matrix[:, 4:]

    # Normalize keypoints
    #norm_kpts = cv2.undistortPoints(norm_kpts.reshape(-1, 1, 2), camera_matrix, dist_coeffs, None, None)
    norm_kpts = norm_kpts.reshape(-1, 2)
    norm_kpts[:, 0] = (norm_kpts[:, 0] - cx) / fx
    norm_kpts[:, 1] = (norm_kpts[:, 1] - cy) / fy

    return norm_kpts

'''
def triangulate_points_dlt(kp1_matched, kp2_matched, P1, P2):
    """
    Triangulate 3D point from 2D correspondences using Direct Linear Transform (DLT)
    """
    
    # Normalize image coordinates
    kp1_norm = np.linalg.inv(P1[:3, :3]) @ np.vstack((kp1_matched.T, np.ones((1, kp1_matched.shape[0]))))
    kp1_norm /= kp1_norm[-1, :]
    
    kp2_norm = np.linalg.inv(P2[:3, :3]) @ np.vstack((kp2_matched.T, np.ones((1, kp2_matched.shape[0]))))
    kp2_norm /= kp2_norm[-1, :]
    
    # Setup homogeneous linear equation Ax = 0
    A = np.zeros((4, 4))
    for i in range(kp1_matched.shape[0]):
        A[0, :] = kp1_norm[:, i].T @ P1[2, :] - P1[0, :]
        A[1, :] = kp1_norm[:, i].T @ P1[2, :] - P1[1, :]
        A[2, :] = kp2_norm[:, i].T @ P2[2, :] - P2[0, :]
        A[3, :] = kp2_norm[:, i].T @ P2[2, :] - P2[1, :]
    
        # Solve for the 3D point using least-squares
        _, _, V = np.linalg.svd(A)
        X_homogeneous = V[-1, :]
        X_homogeneous /= X_homogeneous[-1]
        
    # De-normalize 3D point
    X = np.linalg.inv(P1[:3, :3]) @ X_homogeneous[:3]
    X /= X[-1]
    
    return X
'''




def triangulate_points_opencv(kp1_matched, kp2_matched, intrinsics,rvec_1,rvec_2, tvec_1, tvec_2):
    '''
    function that triangulates points matched between two frames with given intrinsics and camera poses
    
    kp1_matched, kp2_matched:
        (2xN) np array of matched coordinate points between image 1 and 2
    intrinsics:
        (3x3) intrinsics matrix from calibration
    r_vec1, rvec_2, tvec_1, tvec_2:
        (3,) euler angle rotation and translation vectors of two images

    '''
    
    P0, P1 = get_projection_matrices(rvec_1,rvec_2, tvec_1, tvec_2, intrinsics)

    res_1 = cv2.triangulatePoints(P0, P1, kp1_matched, kp2_matched) 
    res_1 = res_1[:3] / res_1[3, :]

    return res_1

def triangulate_points_sksurgery(input_undistorted_points,
                              left_camera_intrinsic_params,
                              right_camera_intrinsic_params,
                              left_to_right_rotation_matrix,
                              left_to_right_trans_vector):
    """
    Function to compute triangulation of points using Harley with cv2.triangulatePoints
    :param input_undistorted_points: [nx4] narray  each input row is [l_x, l_y, r_x, r_y]
    :param left_camera_intrinsic_params: [3x3] narray
    :param right_camera_intrinsic_params: [3x3] narray
    :param left_to_right_rotation_matrix: [3x3] narray
    :param left_to_right_trans_vector: [3x1] narray
    :return output_points: [nx3] narray
    Other related variables:
        left_undistorted, right_undistorted: point image positions in 2 cameras
        left_undistorted[4x2], right_undistorted[4x2] from input_undistorted_points [4x4]
    References
    ----------
    Hartley, Richard I., and Peter Sturm. "Triangulation." Computer vision and image understanding 68, no. 2 (1997): 146-157.
    """

    l2r_mat = stm.construct_rigid_transformation(left_to_right_rotation_matrix, left_to_right_trans_vector)

    # The projection matrix, is just the extrinsic parameters, as our coordinates will be in a normalised camera space.
    # P1 should be identity, so that reconstructed coordinates are in Left Camera Space, to P2 should reflect
    # a right to left transform.
    # Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.
    p1mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.double)

    p2mat = np.zeros((3, 4), dtype=np.double)
    p2mat = l2r_to_p2d(p2mat, l2r_mat)

    number_of_points = input_undistorted_points.shape[0]
    output_points = np.zeros((number_of_points, 3), dtype=np.double)
    # COLORS (RGB)
    output_colors = np.zeros((number_of_points, 3), dtype=np.double)

    # Inverting intrinsic params to convert from pixels to normalised image coordinates.
    k1inv = np.linalg.inv(left_camera_intrinsic_params)
    k2inv = np.linalg.inv(right_camera_intrinsic_params)

    u1_array = np.zeros((3, 1), dtype=np.double)
    u2_array = np.zeros((3, 1), dtype=np.double)

    for dummy_index in range(0, number_of_points):
        u1_array[0, 0] = input_undistorted_points[dummy_index, 0]
        u1_array[1, 0] = input_undistorted_points[dummy_index, 1]
        u1_array[2, 0] = 1

        u2_array[0, 0] = input_undistorted_points[dummy_index, 2]
        u2_array[1, 0] = input_undistorted_points[dummy_index, 3]
        u2_array[2, 0] = 1

        # Converting to normalised image points
        u1t = np.matmul(k1inv, u1_array)
        u2t = np.matmul(k2inv, u2_array)

        # array shapes for input args cv2.triangulatePoints( [3, 4]; [3, 4]; [2, 1]; [2, 1] )
        reconstructed_point = cv2.triangulatePoints(p1mat, p2mat, u1t[:2], u2t[:2])
        reconstructed_point /= reconstructed_point[3]  # Homogenize

        output_points[dummy_index, 0] = reconstructed_point[0]
        output_points[dummy_index, 1] = reconstructed_point[1]
        output_points[dummy_index, 2] = reconstructed_point[2]

        #color = list(image_1[int(x1[1]),int(x1[0]),:]) # always selecting color of first image

        #image[u1_array[0, 0]]
        #output_colors[dummy_index, 0] = reconstructed_point[0] # r
        #output_colors[dummy_index, 1] = reconstructed_point[1] # g
        #output_colors[dummy_index, 2] = reconstructed_point[2] # b

    return output_points


def get_xyz_method_prince(intrinsics, kp1_matched, im1_poses, kp2_matched, im2_poses, image_1=None):


    D3_points = []
    D3_colors = []
    # point in first image
    #pnt1 = np.append(kp1_matched[0],1)
    # pre-multiplying by inverse of camera intrinsics
    #normalised_pnt1 = np.linalg.inv(intrinsics)@pnt1

    # we can use the third row of the equation to get the value of the scaling factor
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    '''

    for idx in range(0,len(kp1_matched)):
        # first calculate X' and Y'
        x1 = np.append(kp1_matched[idx],1) #[u,v,1]_1
        x2 = np.append(kp2_matched[idx],1) #[u,v,1]_2
        # convert to camera coordinates
        x1_norm = np.linalg.inv(intrinsics)@x1
        x2_norm = np.linalg.inv(intrinsics)@x2

        # IMAGE 1 POSES 
        # -------------
        tx, ty, tz, w31, w11, w21, w32, w12, w22, w33, w13, w23 = img_poses_reformat(im1_poses)

        # building equation from prince
        x1_array = np.array([
            [w31*x1_norm[0]-w11,   w32*x1_norm[0]-w12,     w33*x1_norm[0]-w13],
            [w31*x1_norm[1]-w21,   w32*x1_norm[1]-w22,     w33*x1_norm[1]-w23]
        ])

        t1_array = np.array([
            [tx-tz*x1_norm[0]],
            [ty-tz*x1_norm[1]]
        ])

        # IMAGE 2 POSES
        # --------------
        tx, ty, tz, w31, w11, w21, w32, w12, w22, w33, w13, w23 = img_poses_reformat(im2_poses)

        # building equation from prince
        x2_array = np.array([
            [w31*x2_norm[0]-w11,   w32*x2_norm[0]-w12,     w33*x2_norm[0]-w13],
            [w31*x2_norm[1]-w21,   w32*x2_norm[1]-w22,     w33*x2_norm[1]-w23]
        ])

        t2_array = np.array([
            [tx-tz*x2_norm[0]],
            [ty-tz*x2_norm[1]]
        ])

        # combining equations
        A = np.concatenate([x1_array, x2_array])
        b = np.concatenate([t1_array, t2_array])
        #print(f'determinant: {np.linalg.det(A[:3, :])}')

        if np.linalg.det(A[:3, :])==0:
            continue
        else: 
            # solving equations
            X = np.linalg.solve(A[:3,:], b[:3,:])
            D3_points.append(np.ndarray.tolist(X.T[0]))

        #if image_1:
        # solving equations
        color = list(image_1[int(x1[1]),int(x1[0]),:]) # always selecting color of first image
        D3_colors.append(color)
            
    # open3D
    # 
    '''
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig('3D_reconstruction.png')   
    '''

    return D3_points, D3_colors 

def reconstruction_gpt(kp1_matched, kp2_matched, intrinsics):
    
    # Estimate essential matrix from the keypoints
    E, mask = cv2.findEssentialMat(kp1_matched, kp2_matched, intrinsics)
    
    # Decompose the essential matrix into rotation and translation
    R1, R2, t = cv2.decomposeEssentialMat(E)
    
    # Reconstruct the 3D points
    points_3d_homogeneous = cv2.triangulatePoints(intrinsics @ np.hstack((np.identity(3), np.zeros((3, 1)))), 
                                                  intrinsics @ np.hstack((R2, t)),
                                                  kp1_matched.T.reshape(1, -1, 2), 
                                                  kp2_matched.T.reshape(1, -1, 2))
    
    # Convert the homogeneous 3D points to 3D coordinates
    points_3d = points_3d_homogeneous[:3] / points_3d_homogeneous[3]
    
    return points_3d.T


def stereo_rectify_method(image_1, image_2, im1_poses, im2_poses,intrinsics, distortion, imageSize):
    # https://www.youtube.com/watch?v=yKypaVl6qQo

    # relative position between the two is going from the first image to the origin, then from origin to the second image
    T_1_to_2 = np.linalg.inv(im1_poses) @ im2_poses
    # extracting R and T vectors 
    params = extract_rigid_body_parameters(T_1_to_2)

    R = np.array([params[:3]]).T
    T = np.array([params[3:]]).T

    # https://amroamroamro.github.io/mexopencv/matlab/cv.stereoRectify.html
    # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
    rect_L, rect_R, proj_matrix_L, proj_matrix_R, Q, roi_L, roi_R = cv2.stereoRectify(intrinsics, distortion, intrinsics, distortion, imageSize, R, T)
    #R1, R2, P1, P2, Q, roi1, roi2 = S

    # get stereo maps now
    # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
    stereo_map_1_X,stereo_map_1_Y  = cv2.initUndistortRectifyMap(intrinsics, distortion,rect_L,proj_matrix_L, imageSize, cv2.CV_16SC2 )
    stereo_map_2_X, stereo_map_2_Y = cv2.initUndistortRectifyMap(intrinsics, distortion,rect_R,proj_matrix_R, imageSize, cv2.CV_16SC2 )

    # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
    frame_1 = cv2.remap(np.array(image_1), stereo_map_1_X, stereo_map_1_Y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    frame_2 = cv2.remap(np.array(image_2), stereo_map_2_X, stereo_map_2_Y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

    return frame_1, frame_2


def get_xyz(cam1_coords, camera1_M, camera1_R, camera1_T, cam2_coords, camera2_M, camera2_R, camera2_T):
    '''
    camera1_M, camera2_M -> camera matrix intrinsics (3x3) for each
    camera1_R, camera2_R -> camera rotation matrix (3x3) for each
    camera1_T, camera2_T -> (3x1) translation
    camera1_coords, camera2_coords -> (1x2) vector of x and y coordinate in image, and matching one in the other
    '''

    D3_points = []
    for camera1_coords, camera2_coords in zip(cam1_coords, cam2_coords):
        # Get the two key equations from camera1
        camera1_u, camera1_v = camera1_coords
        # Put the rotation and translation side by side and then multiply with camera matrix (extrinsics times intrinsics)
        camera1_P = camera1_M.dot(np.column_stack((camera1_R,camera1_T)))
        # Get the two linearly independent equation referenced in the notes
        camera1_vect1 = camera1_v*camera1_P[2,:]-camera1_P[1,:] # VP3-P2
        camera1_vect2 = camera1_P[0,:] - camera1_u*camera1_P[2,:] #P1-UP3
        
        # Get the two key equations from camera2
        camera2_u, camera2_v = camera2_coords
        # Put the rotation and translation side by side and then multiply with camera matrix
        camera2_P = camera2_M.dot(np.column_stack((camera2_R,camera2_T)))
        # Get the two linearly independent equation referenced in the notes
        camera2_vect1 = camera2_v*camera2_P[2,:]-camera2_P[1,:] #vp3-p2
        camera2_vect2 = camera2_P[0,:] - camera2_u*camera2_P[2,:] #p1-up3
        
        # Stack the 4 rows to create one 4x3 matrix
        full_matrix = np.row_stack((camera1_vect1, camera1_vect2, camera2_vect1, camera2_vect2))
        # The first three columns make up A and the last column is b
        A = full_matrix[:, :3]
        b = full_matrix[:, 3].reshape((4, 1))
        # Solve overdetermined system. Note b in the wikipedia article is -b here.
        # https://en.wikipedia.org/wiki/Overdetermined_system
        soln = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(-b)
        D3_points.append(np.ndarray.tolist(soln.T[0]))
    
    return D3_points


def recover_pose(kp1, kp2, K):
    
    E, mask = cv2.findEssentialMat(kp1.T, kp2.T, focal=1.0, pp=(0., 0.), method=cv2.FM_LMEDS, prob=0.999, threshold=3.0)
    
    points, R, t, mask = cv2.recoverPose(E, kp1.T, kp2.T)
    #R1, R2, t = cv2.decomposeEssentialMat(E)
    
    return R,t

def estimate_camera_poses(kp1_matched, kp2_matched, K):
    #F, _ = cv2.findFundamentalMat(kp1_matched, kp2_matched, cv2.FM_RANSAC)
    '''
    F, _ = cv2.findFundamentalMat(kp1_matched.T, kp2_matched.T, cv2.FM_RANSAC)
    if isinstance(F, np.ndarray):
        pass
    else:
        return None
    '''
    kp1_norm = cv2.undistortPoints(np.expand_dims(kp1_matched, axis=2), K, None)
    kp2_norm = cv2.undistortPoints(np.expand_dims(kp2_matched, axis=2), K, None)

    # Estimate the essential matrix using the normalized points
    E, _ = cv2.findEssentialMat(kp1_norm, kp2_norm, K)

    #E = np.transpose(K) @ F[:3] @ K
    #E = np.matmul(np.matmul(K.T, F), K)

    #E = K.T @ F @ K
    #E = np.matmul(np.matmul(np.transpose(K), F), K)

    _, R, t, _ = cv2.recoverPose(E, kp1_matched.T, kp2_matched.T, K)

    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = np.matmul(np.matmul(U, W), Vt)
    t = U[:, 2]

    # Check for the correct rotation matrix
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    
    return R,t


def method_3( kp1_matched, kp2_matched, K):

    R,t = estimate_camera_poses(kp1_matched, kp2_matched, K)

    p1 = K @ np.hstack((R, t))
    p2 = K @ np.hstack((np.identity(3), np.zeros((3, 1))))

    pts4D = cv2.triangulatePoints(p1, p2, kp1_matched.T, kp2_matched.T)
    pts3D = cv2.convertPointsFromHomogeneous(pts4D.T)


    '''
    # Camera poses (rotation and translation)
    R1 = im1_poses[:3,:3]   # Reference rotation
    t1 = im1_poses[:3,3] # Reference translation

    R2 = im2_poses[:3,:3]   # Reference rotation
    t2 = im2_poses[:3,3] # Reference translation

    
    E = R2 @ t_skew @ np.linalg.inv(R1)
    


    R1_w = R1_cw.T
    t1_w = -R1_cw.T @ t1_cw
    t_skew = np.array([[0, -t1_w[2], t1_w[1]],
                    [t1_w[2], 0, -t1_w[0]],
                    [-t1_w[1], t1_w[0], 0]])
    E = t_skew @ R1_w @ np.linalg.inv(R2) @ np.linalg.inv(t_skew)
    
    points, R, t,mask = cv2.recoverPose(E, kp1, kp2, K)
    '''
    


    
    return pts3D

