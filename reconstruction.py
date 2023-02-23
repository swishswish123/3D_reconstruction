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

def l2r_to_p2d(p2d, l2r):
    """
    Function to convert l2r array to p2d array, which removes last row of l2r to create p2d.
    Notes. l2r_to_p2d() is used in triangulate_points_hartley() to avoid too many variables in one method (see R0914).
    :param p2d: [3x4] narray
    :param l2r: [4x4] narray
    :return p2d: [3x4] narray
    """

    for dummy_row_index in range(0, 3):
        for dummy_col_index in range(0, 4):
            p2d[dummy_row_index, dummy_col_index] = l2r[dummy_row_index, dummy_col_index]

    return p2d

def triangulate_points_opencv(input_undistorted_points,
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

'''
def triangulate_points_opencv_2(kp1_matched, kp2_matched, intrinsics, T_1_to_2):
    
    P1 = intrinsics @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.double)
    P2 = intrinsics @ T_1_to_2[:3,:]

    output_points = cv2.triangulatePoints(P1, P2, kp1_matched.T, kp2_matched.T)

    return (output_points[:3] / output_points[3] ).T
'''

def triangulate_points_opencv_2(kp1_matched, kp2_matched, intrinsics, T_1_to_2):
    
    P1 = intrinsics @ np.hstack((np.identity(3), np.zeros((3, 1))))
    P2 = intrinsics @ T_1_to_2[:3,:]

    kp1_matched = kp1_matched.reshape(-1, 2).T
    kp2_matched = kp2_matched.reshape(-1, 2).T

    # triangulate points
    output_points = cv2.triangulatePoints(P1, P2, kp1_matched, kp2_matched)

    # convert output points to 3D coordinates
    output_points = (output_points / output_points[3])

    return output_points[:3]

def extract_rigid_body_parameters(matrix):
    """
    extract_rigid_body_parameters(matrix)
    extracts parameters from transformation matrix

    Args:
        matrix: 4x4 transformation matrix

    Returns:
        list of all extracted parameters from matrix

    """
    t = matrix[0:3, 3]
    r = matrix[0:3, 0:3]
    rot = spr.from_matrix(r)
    euler = rot.as_euler('zyx', degrees=True)
    return [euler[0], euler[1], euler[2],t[0], t[1], t[2]]


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):

    figsize = (size*2, size*3/4) 
    _, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    for i in range(2):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def read_img(path):
    image = cv2.imread(str(path))
    #w, h = image.shape[1], image.shape[0]
    #w_new, h_new = process_resize(w, h, [640, 480])
    #image = cv2.resize(image.astype('float32'), (w_new, h_new))
    return image

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


def img_poses_reformat(im_poses):
    # MATRICES OF EQUATION 1
    tx = im_poses[0,-1]
    ty = im_poses[1,-1]
    tz = im_poses[2,-1]

    w31 = im_poses[2,0]
    w11 = im_poses[0,0]
    w21 = im_poses[1,0]
    w32 = im_poses[2,1]
    w12 = im_poses[0,1]
    w22 = im_poses[1,1]
    w33 = im_poses[2,2]
    w13 = im_poses[0,2]
    w23 = im_poses[1,2]

    return tx, ty, tz, w31, w11, w21, w32, w12, w22, w33, w13, w23


def get_xyz_method_prince(intrinsics,hand_eye, image_1, kp1_matched, im1_poses,image_2, kp2_matched, im2_poses):



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
        x1 = np.append(kp1_matched[idx],1)
        x2 = np.append(kp2_matched[idx],1)
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
            # plotting point
            color = list(image_1[int(x1[1]),int(x1[0]),:]) # always selecting color of first image
            #ax.scatter(X[0], X[1], X[2], marker='o', color=0)
            D3_colors.append(color)
            D3_points.append(np.ndarray.tolist(X.T[0]))
    # open3D
    # 
    '''
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig('3D_reconstruction.png')   
    '''

    return D3_points, D3_colors 


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


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def method_3( kp1_matched, kp2_matched, K):
    F, mask = cv2.findFundamentalMat(kp1_matched, kp2_matched, cv2.FM_RANSAC)
    if isinstance(F, np.ndarray):
        pass
    else:
        return None

    E = np.transpose(K) @ F[:3] @ K


    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    _, R, t, mask = cv2.recoverPose(E, kp1_matched, kp2_matched, K)

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




def get_matched_keypoints_superglue(pair_match):
    # MATCHES info of the two images
    npz = np.load( pair_match )

    kp1 = npz['keypoints0'] # keypoints in first image
    kp2 = npz['keypoints1'] # keypoints in second image
    matches = npz['matches'] # matches- for each point in kp1, finds match in kp2. If -1-> no match

    # selecting in order the indeces of the matching points
    kp1_matched =  kp1[matches>-1] # selecting all indeces that are matches in im1
    kp2_matched =  kp2[matches[matches>-1]] # selecting points whose indeces are matches in sim2        
    return kp1_matched, kp2_matched

def get_matched_keypoints_sift(img1_original, img2_original):

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # convert 2 images to gray
    img1 = cv2.cvtColor(img1_original, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_original, cv2.COLOR_BGR2GRAY)
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    #feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good_kp1_matched = []
    good_kp2_matched = []
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
            # https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python
            # extracting indexes of matched idx from images
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx
            good_kp1_matched.append(kp1[img1_idx].pt)
            good_kp2_matched.append(kp2[img2_idx].pt)

    # converting to np array
    kp1_matched =  np.asarray(good_kp1_matched) # selecting all indeces that are matches in im1
    kp2_matched =  np.asarray(good_kp2_matched) # selecting points whose indeces are matches in sim2        
    
    return kp1_matched, kp2_matched
        

if __name__=='__main__':

    ########################## PARAMS ###################################
    plot_output = False
    method ='method_3' #sksurgery online 
    project_path = Path(__file__).parent.resolve()
    type='random' # random / phantom / EM_tracker_calib
    # RANDOM, UNDISTORTED: arrow / brain  / checkerboard_test_calibrated / gloves / 
    # RANDOM UNDISTORTED MAC CAM: mac_camera /
    # RANDOM, Distorted: books / points / spinal_section / spinal_section_pink
    # EM_TRACKING_CALIB testing_points /testing_lines
    folder = 'mac_camera'
    frame_rate = 1
    TRACKING = False
    intrinsics = np.loadtxt(f'calibration/mac_calibration/intrinsics.txt')
    distortion = np.loadtxt(f'calibration/mac_calibration/distortion.txt')
    
    matching_method = 'sift' # sift / superglue

    ######################## PERFORMING SUPERGLUE MATCHING ########################################
    if matching_method == 'superglue':
        match_pairs.superglue(type, folder, frame_rate=frame_rate)
        # path where all match pairs located (SUPERGLUE)
        output_dir = f'outputs/match_pairs_{type}_{folder}/'
        # matches
        match_paths = glob.glob(f'{project_path}/{output_dir}/*.npz')

    ########################## LOADING ALL ###################################
    
    # folder where all camera info stored- images and poses
    camera_info = f'{project_path}/assets/{type}/{folder}'
    # images
    frames_pth = sorted(glob.glob(f'{camera_info}/images/*.*'))

    # if we are tracking we need to load camera info
    if TRACKING:
        # camera poses
        poses = np.load(f'{camera_info}/vecs.npy')
        # timestamps
        times = np.load(f'{camera_info}/times.npy')
        # calibration data
        hand_eye = np.load(f'calibration/endoscope_calibration/h2e.npy')

    # path where to save reconstructions
    reconstruction_output = f'{project_path}/reconstructions/{method}/{type}/{folder}'
    if not os.path.isdir(reconstruction_output):
        os.makedirs(reconstruction_output)  
    
    ###################### PERFORM 3D RECONSTRUCTION FOR EACH FRAME######################
    D3_points_all = []
    D3_colors_all = []

    for idx in np.arange(0, len(frames_pth)-frame_rate-1, frame_rate):
    #for idx in range(2,3):
        if idx%10==0:
            print(f'image {idx}')
        # jpg frames path of two matched pairs
        im1_path = frames_pth[idx]
        im2_path = frames_pth[idx+frame_rate]

        # image number (eg. 00000001)- excluding extension
        im1 = im1_path.split('/')[-1][:-4]
        im2 = im2_path.split('/')[-1][:-4]
        # loading images    
        img1_original = cv2.imread(im1_path)  
        img2_original = cv2.imread(im2_path) 
        
        ############################### MATCHING ##########################
        # obtaining or loading keypoints between images 
        if matching_method == 'superglue':
            kp1_matched, kp2_matched = get_matched_keypoints_superglue(f'{project_path}/{output_dir}/{im1}_{im2}_matches.npz')
        elif matching_method == 'sift':
            kp1_matched, kp2_matched = get_matched_keypoints_sift(img1_original, img2_original)
        
        ############################ IMAGE POSES ################################
        if TRACKING:
            # loading poses information of current img pairs
            im1_poses =  poses[idx]  #@ original_point 
            im2_poses =  poses[idx+1] #@ unit_vec

        imageSize = img1_original.shape

        # get color of scatter
        input_undistorted_points = np.concatenate([kp1_matched,kp2_matched],axis=1)
        input_undistorted_points=input_undistorted_points.astype(int) # converting to integer
        D3_colors = img1_original[ input_undistorted_points[:,1],input_undistorted_points[:,0]]
        
        if method=='stereo':
            frame_1, frame_2 = stereo_rectify_method(img1_original, img2_original, im1_poses, im2_poses,intrinsics, distortion, imageSize)
        elif method=='prince':
            #D3_points, D3_colors = get_xyz_method_prince(intrinsics,hand_eye, np.array(img1_original), kp1_matched, im1_poses,np.array(img2_original), kp2_matched, im2_poses)
            D3_points, D3_colors = get_xyz_method_prince(intrinsics,hand_eye, np.array(img1_original), kp1_matched, im1_poses,np.array(img2_original), kp2_matched, im2_poses)

            D3_points_all += D3_points
            D3_colors_all += D3_colors
        elif method == 'gpt':
            R1 = im1_poses[:3,:3]
            R2 = im2_poses[:3,:3]

            t1 = im1_poses[:3,3]
            t2 = im2_poses[:3,3]
            K = intrinsics

            D3_points = reconstruction_gpt(kp1_matched, kp2_matched, K, R1, t1, R2, t2)

            if isinstance(D3_points, np.ndarray):
                    if len(D3_points.shape)>2:
                        D3_points_all += np.ndarray.tolist(D3_points.squeeze())
                    else:
                        D3_points_all += np.ndarray.tolist(D3_points)
                    D3_colors_all += np.ndarray.tolist(D3_colors)
            else:   
                print('no F matrix found') 
        
        elif method == 'method_3':
            D3_points = method_3(kp1_matched, kp2_matched, intrinsics)
            if isinstance(D3_points, np.ndarray):
                if len(D3_points.shape)>2:
                    D3_points_all += np.ndarray.tolist(D3_points.squeeze())
                else:
                    D3_points_all += np.ndarray.tolist(D3_points)
                D3_colors_all += np.ndarray.tolist(D3_colors)
            else:   
                print('no F matrix found') 
        
        elif method=='sksurgery':
            # relative position between the two is going from the first image to the origin, then from origin to the second image
            #T_1_to_2 =   np.linalg.inv(hand_eye) @ im1_poses @ np.linalg.inv(im2_poses) @ hand_eye
            T_1_to_2 =   np.linalg.inv(hand_eye) @ im1_poses @ np.linalg.inv(im2_poses) @ np.linalg.inv(hand_eye)
            # extracting R and T vectors 
            params = extract_rigid_body_parameters(T_1_to_2)

            R = T_1_to_2[:3,:3]
            T = np.array([params[3:]]).T
            '''
            triangulate_points_opencv(input_undistorted_points,
                              left_camera_intrinsic_params,
                              right_camera_intrinsic_params,
                              left_to_right_rotation_matrix,
                              left_to_right_trans_vector)
            '''
            ###### RECTIFY HERE

            if np.size(kp1_matched)==0:
                print('no matches')
            else:
                D3_points = triangulate_points_opencv_2(kp1_matched, kp2_matched, intrinsics, T_1_to_2)

                #D3_points = triangulate_points_opencv(input_undistorted_points, intrinsics, intrinsics, R, T)
                D3_points_all += np.ndarray.tolist(D3_points)
                D3_colors_all += np.ndarray.tolist(D3_colors)
        elif method=='online':

            D3_points = get_xyz(kp1_matched, intrinsics, im1_poses[:3,:3], im1_poses[:3,3:], kp2_matched, intrinsics, im2_poses[:3,:3], im2_poses[:3,3:])
            D3_points_all += D3_points
            # selecting colors from first image
            D3_colors_all += np.ndarray.tolist(D3_colors)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

        '''
        if plot_output:
            image1 = read_img(im1_path)
            image2 = read_img(im2_path)
            plot_image_pair([ image1, image2])

            # match confidence 
            mconf = npz['match_confidence']
            plot_matches(kp1_matched, kp2_matched, color=cm.jet(mconf[matches>-1]))
            #plt.savefig(str(f'{reconstruction_output}/pairs/{im1}_{im2}_matches.png'), bbox_inches='tight', pad_inches=0)
            plt.savefig('matches.png', bbox_inches='tight', pad_inches=0)
        '''
    all_points = np.asarray(D3_points_all)
    all_colors = np.asarray(D3_colors_all)
    
    np.save(f'{reconstruction_output}/points.npy', all_points)
    np.save(f'{reconstruction_output}/colors.npy', all_colors)
    print('done')
    #f.close()