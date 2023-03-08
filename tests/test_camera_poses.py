import cv2
import numpy as np
import unittest
import reconstruction
from reconstruction_utils.utils import get_projection_matrices, eulerAnglesToRotationMatrix, rigid_body_parameters_to_matrix
from reconstruction_utils.reconstruction_algorithms import estimate_camera_poses, triangulate_points_opencv_2, recover_pose
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as spr
import math
import matplotlib.pyplot as plt
from pathlib import Path


def test_camera_poses():

    project_path = Path(__file__).parent.resolve()
    K = np.loadtxt(f'{project_path}/calibration/mac_calibration/intrinsics.txt')
    distortion = np.loadtxt(f'{project_path}/calibration/mac_calibration/distortion.txt')
    
    # 3D world points
    #xyz  = np.random.uniform(low=0, high=K[1,2], size=(100,3))
    # 3D world points
    xyz  = np.random.uniform(low=-1, high=1, size=(200,3))
    xyz[:, :2] *= 0.5*K[1,2]   # scale x and y by half of principal point value
    xyz[:, 2] *= 0.5*(K[0,0] + K[1,1])  # scale z by half of average focal length
    xyz_hom = cv2.convertPointsToHomogeneous(xyz).squeeze()

    # defining extrinsics of two camera views
    rvec_1 = np.zeros(3)
    tvec_1 = np.zeros(3)

    rvec_2 = np.array([10,8,0])
    tvec_2 = np.zeros(3)
    # x,y,z translation of second img
    tvec_2[0] = 1   #x
    tvec_2[1] = 100  #y
    tvec_2[2] = 200   #z

    T_1_to_2 = np.array([
            [1,0,0,tvec_2[0]],
            [0,1,0,tvec_2[1]],
            [0,0,1,tvec_2[2]],
            [0,0,0,1]
        ])
    
    # projecting points
    P0, P1 = get_projection_matrices(rvec_1,rvec_2, tvec_1, tvec_2, K)
    points = xyz_hom.T

    im0 = P0@points
    im0 = im0[:3, :] / im0[2, :]

    im1 = K@ P1@points
    im1 = K@ im1[:3, :] / im1[2, :]

    # projecting points    
    kp1 = im0[:2,:]
    kp2 = im1[:2,:]


    R1,t = recover_pose(kp1, kp2, K)
    #R,t = estimate_camera_poses(image1_points,image2_points, self.intrinsics)
    print('T1 to T2')
    print(rvec_1,rvec_2, tvec_1, tvec_2)
    print('R')
    print(R1.round())
    #print(R2.round())
    print('T')
    print(t.round())


if __name__=='__main__':
    test_camera_poses()