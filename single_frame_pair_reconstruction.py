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
import copy
from reconstruction_utils.utils import *
from reconstruction_utils.reconstruction_algorithms import *

if __name__=='__main__':

    ########################## PARAMS ###################################
    plot_output = False
    method ='sksurgery' #sksurgery online 
    project_path = Path(__file__).parent.resolve()
    
    # CHANGE ME:
    type='tests' # random / phantom / EM_tracker_calib / tests
    folder = 'aruCo'
    # tests: shelf, wardrobe
    # RANDOM, UNDISTORTED: arrow / brain  / checkerboard_test_calibrated / gloves / 
    # RANDOM UNDISTORTED MAC CAM: mac_camera /
    # RANDOM, Distorted: books / points / spinal_section / spinal_section_pink
    # EM_TRACKING_CALIB testing_points /testing_lines
    # PHANTOM: surface / right_in / phantom_surface_2 / both_mid
    
    frame_rate = 1
    TRACKING = True
    intrinsics = np.loadtxt(f'{project_path}/calibration/mac_calibration/intrinsics.txt')
    distortion = np.loadtxt(f'{project_path}/calibration/mac_calibration/distortion.txt')
    
    matching_method = 'sift' #  / superglue
    dist_correction = False

    # path where to save reconstructions
    reconstruction_output = f'{project_path}/reconstructions/{method}/{type}/{folder}'
    if not os.path.isdir(reconstruction_output):
        os.makedirs(reconstruction_output)  
    
    ############### LOAD IMGS
    camera_info = f'{project_path}/assets/{type}/{folder}'
    frames_pth = sorted(glob.glob(f'{camera_info}/images/*.*'))
    
    im1_pth = frames_pth[0]
    im2_pth = frames_pth[1]

    img1 = cv2.imread(im1_pth)  
    img2 = cv2.imread(im2_pth) 

    ###### matches
    if matching_method == 'superglue':
        im1 = im1_pth.split('/')[-1][:-4]
        im2 = im2_pth.split('/')[-1][:-4]
        match_pairs.superglue(type, folder, frame_rate=frame_rate)
        # path where all match pairs located (SUPERGLUE)
        output_dir = f'outputs/match_pairs_{type}_{folder}/'
        # matches
        match_paths = glob.glob(f'{project_path}/{output_dir}/*.npz')
        kp1_matched, kp2_matched = get_matched_keypoints_superglue(f'{project_path}/{output_dir}/{im1}_{im2}_matches.npz')
    elif matching_method == 'sift':
        kp1_matched, kp2_matched = get_matched_keypoints_sift(img1, img2)
    
    #kp1_matched, kp2_matched = get_matched_keypoints_sift(img1, img2)

    #### estimate poses
    #R,t = estimate_camera_poses(kp1_matched.T, kp2_matched.T, intrinsics)
    rvecs = np.load(f'{camera_info}/rvecs.npy')
    tvecs = np.load(f'{camera_info}/tvecs.npy')

    #im1_mat = rigid_body_parameters_to_matrix([rvecs[0],tvecs[0]])
    im1_mat = rigid_body_parameters_to_matrix(np.concatenate([rvecs[0], tvecs[0]]))
    im2_mat = rigid_body_parameters_to_matrix(np.concatenate([rvecs[1], tvecs[1]]))

    im1_poses = rigid_body_parameters_to_matrix([0,0,0,0,0,0])
    
    im2_poses = np.linalg.inv(im1_mat) @ im2_mat # np.linalg.inv(im2_mat) @ im1_mat
    #im2_poses = np.hstack([R,t])
    #im2_poses = np.vstack([im2_poses,np.array([0,0,0,1])])
    
    imageSize = img1.shape

    # get color of scatter
    input_undistorted_points = np.concatenate([kp1_matched,kp2_matched],axis=1)
    input_undistorted_points=input_undistorted_points.astype(int) # converting to integer
    D3_colors = img1[ input_undistorted_points[:,1],input_undistorted_points[:,0]]
    
    params = extract_rigid_body_parameters(im2_poses)
    print(params)
    rvec_2 = params[:3]
    tvec_2 = params[3:]
    D3_points  = triangulate_points_opencv_2(kp1_matched.T, kp2_matched.T, intrinsics,[0,0,0],rvec_2, [0,0,0], tvec_2)
    #D3_colors = D3_colors[mask]
    #D3_points = triangulate_points_opencv(input_undistorted_points, intrinsics, intrinsics, R, T)
    #D3_points_all += np.ndarray.tolist(D3_points.T)
    #D3_colors_all += np.ndarray.tolist(D3_colors)

    all_points = np.asarray(D3_points, dtype='float').T
    all_colors = np.asarray(D3_colors, dtype='float')
    
    np.save(f'{reconstruction_output}/points.npy', all_points)
    np.save(f'{reconstruction_output}/colors.npy', all_colors)
    print('done')

    '''
    HEst = calcBestHomography(kp1_matched, kp2_matched)
    Hcv, _= cv2.findHomography(kp1_matched, kp2_matched)

    # deompose to get R and t
    H = HEst.T
    h1 = H[0]
    h2 = H[1]
    h3 = H[2]
    K_inv = np.linalg.inv(intrinsics)
    L = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = L * np.dot(K_inv, h1)
    r2 = L * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)
    T = L * (K_inv @ h3.reshape(3, 1))
    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))
    '''


