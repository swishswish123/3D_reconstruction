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


def interpolate_between_lines(kp1_matched,  kp2_matched):
    # 0->1
    # 1->5
    # 5->3
    num_interp_pnts = 10
    interp_between=[[1,0],[5,4], [4,3], [1,5]] # , 
    new_kp1 = copy.deepcopy(kp1_matched)
    new_kp2 = copy.deepcopy(kp2_matched)

    for point in interp_between:
        x1_start, y1_start = kp1_matched[point[0]]
        x1_end, y1_end = kp1_matched[point[1]]

        x2_start, y2_start = kp2_matched[point[0]]
        x2_end, y2_end = kp2_matched[point[1]]

        # creating x values to interpolate
        x1 = np.linspace(x1_start, x1_end, num_interp_pnts)[1:-1]
        # interpolating to get y of these x points
        y1 = np.interp(x1, [x1_start,x1_end],[y1_start,y1_end])
        # stacking and combining
        new_kp1 = np.concatenate([new_kp1, np.vstack([x1,y1]).T])

        # creating x values to interpolate
        x2 = np.linspace(x2_start, x2_end, num_interp_pnts)[1:-1]
        # interpolating to get y of these x points
        y2 = np.interp(x2, [x2_start,x2_end],[y2_start,y2_end])
        # stacking and combining
        new_kp2 = np.concatenate([new_kp2, np.vstack([x2,y2]).T])

    plt.figure()
    plt.scatter(new_kp1[:,0], new_kp1[:,1])
    plt.scatter(new_kp2[:,0], new_kp2[:,1])
    plt.savefig('testing.png')

if __name__=='__main__':

    ########################## PARAMS ###################################
    plot_output = False
    method ='sksurgery' #sksurgery online 
    project_path = Path(__file__).parent.resolve()
    
    # CHANGE ME:
    type='EM_tracker_calib' # random / phantom / EM_tracker_calib
    folder = 'testing_lines'
    # RANDOM, UNDISTORTED: arrow / brain  / checkerboard_test_calibrated / gloves / 
    # RANDOM UNDISTORTED MAC CAM: mac_camera /
    # RANDOM, Distorted: books / points / spinal_section / spinal_section_pink
    # EM_TRACKING_CALIB testing_points /testing_lines
    # PHANTOM: surface / right_in / phantom_surface_2 / both_mid
    
    frame_rate = 1
    TRACKING = 'EM'
    
    intrinsics = np.loadtxt(f'{project_path}/calibration/endoscope_calibration/intrinsics.txt')
    distortion = np.loadtxt(f'{project_path}/calibration/endoscope_calibration/distortion.txt')
    
    matching_method = 'superglue' # sift / superglue
    dist_correction = False
    
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
        # loading images    ############################################
        img1_original = cv2.imread(im1_path)  
        img2_original = cv2.imread(im2_path) 
        
        if dist_correction:
            # only do this the first time
            h,  w = img1_original.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics,distortion,(w,h),1,(w,h))
            
            # undistorting images
            
            #mapx,mapy = cv2.initUndistortRectifyMap(intrinsics,distortion,None,newcameramtx,(w,h),5)
            #dst1 = cv2.remap(img1_original,mapx,mapy,cv2.INTER_LINEAR)

            dst1 = cv2.undistort(copy.deepcopy(img1_original), intrinsics, distortion, None, newcameramtx)
            dst2 = cv2.undistort(copy.deepcopy(img2_original) , intrinsics, distortion, None, newcameramtx)

            x, y, w, h = roi
            img1_original = dst1[y:y+h, x:x+w]
            img2_original = dst2[y:y+h, x:x+w]

            
            #dst2 = cv2.remap(img2_original,mapx,mapy,cv2.INTER_LINEAR)

            #undistorted = undistorted[y:y+h, x:x+w]
            #undistorted_2 = undistorted_2[y:y+h, x:x+w]
            numpy_vertical = np.concatenate((cv2.resize(dst1,(w,h)), cv2.resize(img1_original,(w,h))))
            cv2.imshow('',numpy_vertical)
        
        ############################### MATCHING ##########################
        # obtaining or loading keypoints between images 
        if matching_method == 'superglue':
            kp1_matched, kp2_matched = get_matched_keypoints_superglue(f'{project_path}/{output_dir}/{im1}_{im2}_matches.npz')
        elif matching_method == 'sift':
            kp1_matched, kp2_matched = get_matched_keypoints_sift(img1_original, img2_original)
        
        ############################ IMAGE POSES ################################
        if TRACKING=='EM':
            # loading poses information of current img pairs
            im1_poses =  poses[idx]  #@ original_point 
            im2_poses =  poses[idx+frame_rate] #@ unit_vec
        elif TRACKING == 'aruCo':
            im0_mat = rigid_body_parameters_to_matrix(np.concatenate([rvecs[0], tvecs[0]]))
            im1_mat = rigid_body_parameters_to_matrix(np.concatenate([rvecs[idx], tvecs[idx]]))
            im2_mat = rigid_body_parameters_to_matrix(np.concatenate([rvecs[idx+frame_rate], tvecs[idx+frame_rate]]))

            im1_mat = im1_mat @  np.linalg.inv(im0_mat) 
            im2_mat =  im2_mat @ np.linalg.inv(im0_mat) 
        else:
            R,t = estimate_camera_poses(kp1_matched.T, kp2_matched.T, intrinsics)
            im1_poses = rigid_body_parameters_to_matrix([0,0,0,0,0,0])
            im2_poses = np.hstack([R,t])
            im2_poses = np.vstack([im2_poses,np.array([0,0,0,1])])

        imageSize = img1_original.shape

        # get color of scatter
        input_undistorted_points = np.concatenate([kp1_matched,kp2_matched],axis=1)
        input_undistorted_points=input_undistorted_points.astype(int) # converting to integer
        D3_colors = img1_original[ input_undistorted_points[:,1],input_undistorted_points[:,0]]
        
        if method=='stereo':
            frame_1, frame_2 = stereo_rectify_method(img1_original, img2_original, im1_poses, im2_poses,intrinsics, distortion, imageSize)
        elif method=='prince':
            #D3_points, D3_colors = get_xyz_method_prince(intrinsics,hand_eye, np.array(img1_original), kp1_matched, im1_poses,np.array(img2_original), kp2_matched, im2_poses)
            D3_points, D3_colors = get_xyz_method_prince(intrinsics, kp1_matched, im1_poses, kp2_matched, im2_poses, image_1=np.array(img1_original))

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
            #T_1_to_2 =   np.linalg.inv(hand_eye) @ im1_poses @ np.linalg.inv(im2_poses) @ np.linalg.inv(hand_eye)
            #T_1_to_2 =  np.linalg.inv(im2_poses) @ im1_poses

            if TRACKING=='EM':
                T_1_to_2 =  hand_eye@np.linalg.inv(im2_poses) @ im1_poses @  np.linalg.inv(hand_eye)
            else:
                im1_poses = rigid_body_parameters_to_matrix([0,0,0,0,0,0])
                T_1_to_2 = np.linalg.inv(im1_mat) @ im2_mat # np.linalg.inv(im2_mat) @ im1_mat
            #T_1_to_2 = hand_eye@np.linalg.inv(im2_poses) @ im1_poses@np.linalg.inv(hand_eye)

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
                #D3_points  = triangulate_points_opencv_2(kp1_matched, kp2_matched, intrinsics, T_1_to_2, im1_poses, im2_poses,  distortion)
                
                rvec_1 = np.zeros(3)
                tvec_1 =  np.zeros(3)
                params = extract_rigid_body_parameters(T_1_to_2)
                rvec_2 = params[:3]
                tvec_2 = params[3:]
                D3_points  = triangulate_points_opencv_2(kp1_matched.T, kp2_matched.T, intrinsics,rvec_1,rvec_2, tvec_1, tvec_2)
                #D3_colors = D3_colors[mask]
                #D3_points = triangulate_points_opencv(input_undistorted_points, intrinsics, intrinsics, R, T)
                D3_points_all += np.ndarray.tolist(D3_points.T)
                D3_colors_all += np.ndarray.tolist(D3_colors)
        elif method=='online':

            D3_points = get_xyz(kp1_matched, intrinsics, im1_poses[:3,:3], im1_poses[:3,3:], kp2_matched, intrinsics, im2_poses[:3,:3], im2_poses[:3,3:])
            D3_points_all += D3_points
            # selecting colors from first image
            D3_colors_all += np.ndarray.tolist(D3_colors)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
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
    all_points = np.asarray(D3_points_all, dtype='float')
    all_colors = np.asarray(D3_colors_all, dtype='float')
    
    np.save(f'{reconstruction_output}/points.npy', all_points)
    np.save(f'{reconstruction_output}/colors.npy', all_colors)
    print('done')
    #f.close()