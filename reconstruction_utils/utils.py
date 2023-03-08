
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
import math



def solveAXEqualsZero(A):
    # TO DO: Write this routine - it should solve Ah = 0. You can do this using SVD. Consult your notes! 
    # Hint: SVD will be involved. 
    
    _,_,V = np.linalg.svd(A)
    h = V.T[:,-1]
  
    return h

def calcBestHomography(pts1Cart, pts2Cart):
    
    # This function should apply the direct linear transform (DLT) algorithm to calculate the best 
    # homography that maps the cartesian points in pts1Cart to their corresonding matching cartesian poitns 
    # in pts2Cart.
    
    # This function calls solveAXEqualsZero. Make sure you are wary of how to reshape h into a 3 by 3 matrix. 
    
    n_points = pts1Cart.shape[1]
    
    # TO DO: replace this:
    # H = np.identity(3)

    # TO DO: 
    # First convert points into homogeneous representation
    # Hint: we've done this before  in the skeleton code we provide.
    pts1Hom = np.concatenate((pts1Cart, np.ones((1,pts1Cart.shape[1]))), axis=0)
    pts2Hom = np.concatenate((pts2Cart, np.ones((1,pts2Cart.shape[1]))), axis=0)
    
    # Then construct the matrix A, size (n_points * 2, 9)
    A = np.zeros((n_points*2, 9))
    
    row_num = 0
    
    # Consult the notes!
    for idx in range(n_points):
        u = pts1Hom[0,idx]
        v = pts1Hom[1,idx]
        w = pts1Hom[2,idx]
        
        x = pts2Hom[0,idx]
        y = pts2Hom[1,idx]
        z = pts2Hom[2,idx]
        
        first_row = np.array( [0, 0, 0, -u, -v, -w,   y*u,  y*v,  y])
        second_row = np.array([u, v, w,  0,  0,  0,  -x*u, -x*v, -x])
         
        A[row_num,:] = first_row
        row_num+=1
        A[row_num,:] = second_row
        row_num+=1
        
    # Solve Ah = 0 using solveAXEqualsZero and get h.
    h = solveAXEqualsZero(A)
    
    # Reshape h into the matrix H, values of h go first into rows of H
    H = h.reshape((3, 3))
    
    return H

def un_normalise(kp_norm, intrinsics):
    kp_hom = cv2.convertPointsToHomogeneous(kp_norm).squeeze()
    kp_hom= kp_hom@intrinsics
    kp_orig = cv2.convertPointsFromHomogeneous(kp_hom).squeeze()
    return kp_orig

def normalise(kp_matched, intrinsics):
    kp_hom = cv2.convertPointsToHomogeneous(kp_matched).squeeze()
    #kp_matched = np.matmul(kp_hom, np.linalg.inv(intrinsics))
    kp_matched_hom = kp_hom @ np.linalg.inv(intrinsics)
    
    kp_matched = cv2.convertPointsFromHomogeneous(kp_matched_hom).squeeze()
    return kp_matched


def rigid_body_parameters_to_matrix(params):
    """
    rigid_body_parameters_to_matrix(params)
    converts a list of rigid body parameters to transformation matrix

    Args:
        params: list of rigid body parameters

    Returns:
        4x4 transformation matrix of these parameters

    """
    matrix = np.eye(4)
    r = (spr.from_euler('zyx', [params[0], params[1], params[2]], degrees=True)).as_matrix()
    matrix[0:3, 0:3] = r
    matrix[0][3] = params[3]
    matrix[1][3] = params[4]
    matrix[2][3] = params[5]
    return matrix

def eulerAnglesToRotationMatrix( theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])    

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],  
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0], 
                    [math.sin(theta[2]),    math.cos(theta[2]),     0], 
                    [0,                     0,                      1]
                    ])    

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R



def get_projection_matrices(rvec_1,rvec_2, tvec_1, tvec_2, K):
    #hom_xyz = cv2.convertPointsToHomogeneous(self.xyz).squeeze().T

    T0 = np.array(tvec_1) # Translation vector
    R0 = eulerAnglesToRotationMatrix(rvec_1) # Rotation matrix:
    RT0 = np.zeros((3,4))  # combined Rotation/Translation matrix
    RT0[:3,:3] = R0
    RT0[:3, 3] = T0
    P0 = K@RT0 # Projection matrix

    T1 = np.array(tvec_2)
    R1 = eulerAnglesToRotationMatrix(rvec_2) # Rotation matrix:
    RT1 = np.zeros((3,4))
    RT1[:3,:3] = R1
    RT1[:3, 3] = -T1
    P1 = K@ RT1
    return P0, P1


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


'''
def triangulate_points_opencv_2(kp1_matched, kp2_matched, intrinsics, T_1_to_2):
    
    P1 = intrinsics @ np.hstack((np.identity(3), np.zeros((3, 1))))
    P2 = intrinsics @ T_1_to_2[:3,:]

    kp1_matched = kp1_matched.reshape(-1, 2).T
    kp2_matched = kp2_matched.reshape(-1, 2).T

    # triangulate points
    output_points = cv2.triangulatePoints(P1, P2, kp1_matched, kp2_matched*0.001)

    # convert output points to 3D coordinates
    output_points = (output_points / output_points[3])

    return output_points[:3]

'''


def eulerAnglesToRotationMatrix( theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])    

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],  
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0], 
                    [math.sin(theta[2]),    math.cos(theta[2]),     0], 
                    [0,                     0,                      1]
                    ])    

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

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


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


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
    #orb = cv2.ORB_create()

    # convert 2 images to gray
    img1 = cv2.cvtColor(img1_original, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_original, cv2.COLOR_BGR2GRAY)
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # find the keypoints and descriptors with ORB
    #kp1, des1 = orb.detectAndCompute(img1,None)
    #kp2, des2 = orb.detectAndCompute(img2,None)

    #feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    '''
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    '''
    # Apply ratio test
    good_kp1_matched = []
    good_kp2_matched = []
    good = []
    
    for m,n in matches:
        
        if m.distance < 0.5*n.distance:
            good.append(m)
            # https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python
            # extracting indexes of matched idx from images
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx
            good_kp1_matched.append(kp1[img1_idx].pt)
            good_kp2_matched.append(kp2[img2_idx].pt)
    '''
    for m in matches:
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
        good_kp1_matched.append(kp1[img1_idx].pt)
        good_kp2_matched.append(kp2[img2_idx].pt)
    '''
    # converting to np array
    kp1_matched =  np.asarray(good_kp1_matched) # selecting all indeces that are matches in im1
    kp2_matched =  np.asarray(good_kp2_matched) # selecting points whose indeces are matches in sim2        
    
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, img2, flags=2)
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.savefig('plot.png')
    return kp1_matched, kp2_matched
        


