# this file will perform a sparse 3D reconstruction. The output will be 3D points and colors (RGB)
# you only need to adjust any variables under 'utils'

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
from reconstruction_utils.utils import interpolate_between_lines

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

def match_features(img1, img2):
    # Plot images side by side
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    plt.show(block=False)

    # Select matching points
    matches = []
    fig.canvas.set_window_title('Click on matching points. Close the window when finished.')
    for i in range(4):
        fig.canvas.manager.window.activateWindow()
        fig.canvas.manager.window.raise_()
        pts = plt.ginput(n=1, timeout=0, show_clicks=True)
        matches.append(pts)

    # Convert matching points to pixel coordinates
    matches = np.array(matches)
    matches = matches.astype(int)

    # Save matches as np arrays
    #np.save('matches_img1.npy', matches[0])
    #np.save('matches_img2.npy', matches[1])

    return matches


def select_matches(img1, img2):



    # Define a callback function for mouse clicks
    matches = []

    def onclick(event):
        # Check which axes was clicked
        if event.inaxes == ax[0]:
            print('point 1')
            matches.append([event.xdata, event.ydata, None, None])
            ax[0].scatter(event.xdata, event.ydata, marker='+', s=100, color='red')
            plt.draw()
        elif event.inaxes == ax[1]:
            print('point 2')
            matches[-1][2:] = [event.xdata, event.ydata]
            ax[1].scatter(event.xdata, event.ydata, marker='+', s=100, color='red')
            plt.draw()

    #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Attach the callback function to the figure
    #fig.canvas.mpl_connect('button_press_event', onclick)

    # Wait for the user to close the plot
    # Display the images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #mpl.use('TkAgg')
    ax[0].imshow(img1)
    ax[1].imshow(img2)


    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    #fig.canvas.mpl_connect('button_press_event', onclick)

    # Convert the matches to numpy arrays
    matches = np.array(matches)

    #[381, 500] [192,324]
    #[1825,500] [1282,580]
    # 500,847]  [290, 608]
    # [1705, 815] [1184, 993]
    # [708,673]   [372,500]
    #[708,488] [323, 346]
    # [975,482], [497, 368]
    #
    return matches[:, :2], matches[:, 2:]

if __name__=='__main__':
    '''
    method leave as 'opencv'

    type is the parent folder of the data (right under assets)
    folder is the name of the folder right under assets but on top of your image folder
    so it would be:
    
    superglue/assets/{type}/{folder}/images

    frame rate leave as 1
    
    tracking leave as aruCo
    
    
    '''
    project_path = Path(__file__).parent.resolve()
    mpl.use('TkAgg')

    ########################## PARAMS ###################################
    # method of performing 3D reconstruction
    method ='opencv' # opencv/sksurgery/online/prince
    # parent folder of data type
    type='aruco' # random / phantom / EM_tracker_calib /tests
    # folder right on top of images
    folder = 'shelves_2'
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
    TRACKING = 'aruCo' # EM / aruCo
    
    # change to the correct folder where intrinsics and distortion located
    intrinsics = np.loadtxt(f'{project_path}/calibration/mac_calibration/intrinsics.txt')
    distortion = np.loadtxt(f'{project_path}/calibration/mac_calibration/distortion.txt')
    
    # method of getting keypoints and matching between frames
    # sift- sift for feature matching
    # superglue- uses superglue for feature matching
    # ground_truth- you will manually label corresponding points between frames
    matching_method = 'superglue' # sift / superglue /ground_truth
    
    # perform distortion correction on image or not
    dist_correction = False
    
    ######################## PERFORMING SUPERGLUE MATCHING ########################################
    # if superglue is selected, superglue matching is performed before running and output is saved under outputs
    if matching_method == 'superglue':
        # performing superglue matching
        match_pairs.superglue(type, folder, frame_rate=frame_rate)
        
        # matches paths once performed
        # path where all match pairs located (SUPERGLUE)
        output_dir = f'{project_path}/outputs/match_pairs_{type}_{folder}'
        # matches
        match_paths = glob.glob(f'{output_dir}/*.npz')

    ########################## LOADING ALL ###################################
    
    # folder where all camera info stored- images and poses
    camera_info = f'{project_path}/assets/{type}/{folder}'
    # images
    frames_pth = sorted(glob.glob(f'{camera_info}/images/*.*'))

    # if we are tracking we need to load camera info
    if TRACKING=='EM':
        # camera poses
        poses = np.load(f'{camera_info}/vecs.npy')
        # timestamps
        times = np.load(f'{camera_info}/times.npy')
        # calibration data
        hand_eye = np.load(f'calibration/endoscope_calibration/h2e.npy')
    
    elif TRACKING=='aruCo':
        rvecs = np.load(f'{camera_info}/rvecs.npy')
        tvecs = np.load(f'{camera_info}/tvecs.npy')

    # path where to save reconstructions
    reconstruction_output = f'{project_path}/reconstructions/{method}/{type}/{folder}'
    if not os.path.isdir(reconstruction_output):
        os.makedirs(reconstruction_output)  
    
    ###################### PERFORM 3D RECONSTRUCTION FOR EACH FRAME######################
    D3_points_all = []
    D3_colors_all = []

    for idx in np.arange(0, len(frames_pth)-1, frame_rate):
    #for idx in [0]:
        if idx%10==0:
            print(f'image {idx}')
        # jpg frames path of two matched pairs
        ##################### 1)))))))))))))))))
        im1_path = frames_pth[idx]
        #im1_path = frames_pth[0]
        im2_path = frames_pth[idx+frame_rate]

        # image number (eg. 00000001)- excluding extension
        im1 = im1_path.split('/')[-1][:-4]
        im2 = im2_path.split('/')[-1][:-4]
        # loading images    ############################################
        img1_original = cv2.imread(im1_path)  
        img2_original = cv2.imread(im2_path) 

        ############################### MATCHING ##########################
        # obtaining or loading keypoints between images 
        # kpn_matched will be (N*2)
        if matching_method == 'superglue':
            kp1_matched, kp2_matched = get_matched_keypoints_superglue(f'{output_dir}/{im1}_{im2}_matches.npz')
        elif matching_method == 'sift':
            kp1_matched, kp2_matched = get_matched_keypoints_sift(img1_original, img2_original)
        elif matching_method =='ground_truth':
            kp1_matched, kp2_matched = select_matches(img1_original, img2_original)
            print(kp1_matched, kp2_matched)
        # if no matches found skip to next frame pair, otherwise undistort kpts
        if np.size(kp1_matched)==0:
            print('no matches')
            continue

        ####################### GETTING COLOR OF POINTS ###############################################
        # selecting colors of keypoints which will be triangulated
        input_undistorted_points = np.concatenate([kp1_matched, kp2_matched], axis=1)
        input_undistorted_points = input_undistorted_points.astype(int)  # converting to integer
        D3_colors = img1_original[input_undistorted_points[:, 1], input_undistorted_points[:,0]]  # (Nx3)- where N is same as kp_matched N

        # undistort keypoints
        kp1_matched = cv2.undistortPoints(kp1_matched.astype('float32'), intrinsics, distortion, None,
                                                             intrinsics)
        kp2_matched = cv2.undistortPoints(kp2_matched.astype('float32'), intrinsics, distortion, None,
                                                             intrinsics)

        kp1_matched = kp1_matched.squeeze(1).T # (2xN)
        kp2_matched = kp2_matched.squeeze(1).T # (2xN)

        ############################ IMAGE POSES ################################
        # rotation and translation vec of first frame will be set to zero, R and t of second 
        # camera position will be set to whatever the transform between the two cameras is
        # - calculated below (T_1_to_T_2)
        rvec_1 = np.zeros(3)
        tvec_1 =  np.zeros(3)
        if TRACKING=='EM':
            # selecting poses information of current img pairs
            ################## 2))))))))))))))
            im1_poses = poses[idx]
            #im1_poses = poses[0]
            im2_poses =  poses[idx+frame_rate] 
            # getting relative transform between two camera poses
            # relative position between the two is going from the first image to the origin,
            # then from origin to the second image
            T_1_to_2 =  hand_eye@np.linalg.inv(im2_poses) @ im1_poses @  np.linalg.inv(hand_eye)

        elif TRACKING == 'aruCo':
            # selecting poses of aruco of current frame pair and converting to 4x4 matrix
            ################## 3)))))))))))))))))))
            im1_mat = extrinsic_vecs_to_matrix(rvecs[idx], tvecs[idx]) #(4x4)
            im2_mat = extrinsic_vecs_to_matrix(rvecs[idx+frame_rate], tvecs[idx+frame_rate]) #(4x4)
            # relative transform between two camera poses
            T_1_to_2 =   np.linalg.inv(im2_mat) @ im1_mat # (4x4)

            # rotation and translation vectors between two frames in euler angles
            rvec_2, tvec_2 = extrinsic_matrix_to_vecs(T_1_to_2)
        else:
            # estimating camera poses
            R,t = estimate_camera_poses(kp1_matched, kp2_matched, intrinsics)
            im1_poses = extrinsic_vecs_to_matrix([0,0,0],[0,0,0])
            im2_poses = np.hstack([R,t])
            im2_poses = np.vstack([im2_poses,np.array([0,0,0,1])])

        imageSize = img1_original.shape
        

        ######################### TRIANGULATION- GETTING 3D POINTS ##################################
        if method == 'opencv':
            '''
            triangulate with opencv function
            '''
            
            D3_points = triangulate_points_opencv(kp1_matched, kp2_matched, intrinsics, rvec_1, rvec_2, tvec_1, tvec_2) # (3xN)
            # saving 3D points and colours to array of all pointcloud
            D3_points_all += np.ndarray.tolist(D3_points) # (NX3)
            D3_colors_all += np.ndarray.tolist(D3_colors) # (NX3)

        elif method=='prince':
            '''
            triangulating with prince method 
            '''
            D3_points, D3_colors = get_xyz_method_prince(intrinsics, kp1_matched, im1_poses, kp2_matched, im2_poses, image_1=np.array(img1_original))

            D3_points_all += D3_points
            D3_colors_all += D3_colors
        
        elif method=='sksurgery':

            # extracting R and T vectors 
            r, t = extrinsic_matrix_to_vecs(T_1_to_2)
            R = T_1_to_2[:3,:3]
            T = np.array([t[3:]]).T
            D3_points = triangulate_points_sksurgery(input_undistorted_points, intrinsics, intrinsics, R, T)
            D3_points_all += np.ndarray.tolist(D3_points.T)
            D3_colors_all += np.ndarray.tolist(D3_colors)
        elif method=='online':

            D3_points = get_xyz(kp1_matched, intrinsics, im1_poses[:3,:3], im1_poses[:3,3:], kp2_matched, intrinsics, im2_poses[:3,:3], im2_poses[:3,3:])
            D3_points_all += D3_points
            # selecting colors from first image
            D3_colors_all += np.ndarray.tolist(D3_colors)
        
        elif method == 'method_3':
            '''
            with camera pose estimation
            '''
            D3_points = method_3(kp1_matched, kp2_matched, intrinsics)
            if isinstance(D3_points, np.ndarray):
                if len(D3_points.shape)>2:
                    D3_points_all += np.ndarray.tolist(D3_points.squeeze())
                else:
                    D3_points_all += np.ndarray.tolist(D3_points)
                D3_colors_all += np.ndarray.tolist(D3_colors)
            else:   
                print('no F matrix found') 
        elif method=='stereo':
            '''
            stereo rectification method- align images
            '''
            frame_1, frame_2 = stereo_rectify_method(img1_original, img2_original, im1_poses, im2_poses,intrinsics, distortion, imageSize)
        

        if cv2.waitKey(1) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            break

    # convert points to array for all point cloud
    all_points = np.asarray(D3_points_all, dtype='float')
    all_colors = np.asarray(D3_colors_all, dtype='float')
    
    # output saved as (NX3) points numpy array- 3 dimensions representing X,Y,Z
    np.save(f'{reconstruction_output}/points.npy', all_points)
    # output saved as (NX3) points numpy array- 3 dimensions representing R,G,B
    np.save(f'{reconstruction_output}/colors.npy', all_colors)
    print('done')
