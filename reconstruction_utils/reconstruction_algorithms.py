

from .utils import get_projection_matrices, img_poses_reformat, extrinsic_matrix_to_vecs, estimate_camera_poses

import numpy as np
import cv2


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

    return res_1.T



def get_xyz_method_prince(intrinsics, kp1_matched, im1_poses, kp2_matched, im2_poses):


    D3_points = []

    # we can use the third row of the equation to get the value of the scaling factor

    color_mask = []

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

        # obtaining colour
        #color = list(image_1[int(x1[1]),int(x1[0]),:]) # always selecting color of first image
        color_mask.append(idx)

    return D3_points, np.array(color_mask )


def stereo_rectify_method(image_1, image_2, im1_poses, im2_poses,intrinsics, distortion, imageSize):
    # https://www.youtube.com/watch?v=yKypaVl6qQo

    # relative position between the two is going from the first image to the origin, then from origin to the second image
    T_1_to_2 = np.linalg.inv(im1_poses) @ im2_poses
    # extracting R and T vectors 
    params = extrinsic_matrix_to_vecs(T_1_to_2)

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

