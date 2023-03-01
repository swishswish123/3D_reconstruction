import unittest
import reconstruction
from reconstruction_utils.utils import get_projection_matrices, eulerAnglesToRotationMatrix, rigid_body_parameters_to_matrix
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as spr
import math
import matplotlib.pyplot as plt
from pathlib import Path

class TestReconstruction(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        project_path = Path(__file__).parent.resolve()
        self.intrinsics = np.loadtxt(f'{project_path}/calibration/mac_calibration/intrinsics.txt')
        self.distortion = np.loadtxt(f'{project_path}/calibration/mac_calibration/distortion.txt')
         
        # defining 3d world coordinate points
        self.xyz  = np.array([[940, 530, 50],
                              [5.0, 10., 20.0],
                              [500.0, 110., 20.0],
                              [56.0, 0.8, 21.0]
                                ]) # define an arbitrary 3D point in world coordinates
        # converting to homogenous
        self.xyz_hom = cv2.convertPointsToHomogeneous(self.xyz).squeeze()

        print('original xyz')
        print(f'{self.xyz_hom.round()}')
        
        # defining extrinsics of two camera views
        self.rvec_1 = np.zeros(3)
        self.tvec_1 = np.zeros(3)

        self.rvec_2 = np.zeros(3)
        self.tvec_2 = np.zeros(3)
        # x,y,z translation of second img
        self.tvec_2[0] = 0   #x
        self.tvec_2[1] = 10  #y
        self.tvec_2[2] = 2   #z


        self.T_1_to_2 = np.array([
            [1,0,0,self.tvec_2[0]],
            [0,1,0,self.tvec_2[1]],
            [0,0,1,self.tvec_2[2]],
            [0,0,0,1]
        ])

        return

    def tearDown(self):
        pass
    

    def test_projection_matrices(self):
        # testing if projection using opencv method and proj matrices same
        # OPENCV
        image1_points, _ = cv2.projectPoints(self.xyz, self.rvec_1, self.tvec_1, self.intrinsics, self.distortion)
        image1_points = image1_points.squeeze()      

        image2_points, _ = cv2.projectPoints(self.xyz, self.rvec_2, self.tvec_2, self.intrinsics, self.distortion)
        image2_points = image2_points.squeeze()

        # Projection wa
        P0, P1 = get_projection_matrices(self.rvec_1,self.rvec_2, self.tvec_1, self.tvec_2, self.intrinsics)
        
        #####################
        for point in self.xyz_hom:
            im_point_0 = P0 @ point # project this point using the first camera pose
            im_point_0 = im_point_0/im_point_0[-1] # normalize as we are in homogenuous coordinates

            im_point_1 = P1 @ point
            im_point_1 = im_point_1/im_point_1[-1] # normalize as we are in homogenuous coordinates
            

        plt.figure()
        plt.scatter(self.image1_points[:,0], self.image1_points[:,0])
        plt.scatter(im_point_0[0], im_point_1[1])
        plt.savefig("mygraph.png")



    def test_projection(self):

        P0, P1 = get_projection_matrices(self.rvec_1,self.rvec_2, self.tvec_1, self.tvec_2, self.intrinsics)
        
        for point in self.xyz_hom:
            # Projecting 3D point to 2D using projection matrices
            im_point_0 = P0 @ point # project this point using the first camera pose
            im_point_0 = im_point_0/im_point_0[-1] # normalize as we are in homogenuous coordinates

            im_point_1 = P1 @ point
            im_point_1 = im_point_1/im_point_1[-1] # normalize as we are in homogenuous coordinates
           

            res = cv2.triangulatePoints(P0, P1, im_point_0[:2], im_point_1[:2]) 
            res = res[:3]/res[-1]

            

    
    def test_triangulation_opencv_2(self):
        im1_poses = rigid_body_parameters_to_matrix(np.concatenate((self.rvec_1,self.tvec_1)))
        im2_poses = rigid_body_parameters_to_matrix(np.concatenate((self.rvec_2,self.tvec_2)))
        #im1_norm = np.matmul(np.linalg.inv(self.intrinsics), self.image1_points)
        #im2_norm = np.matmul(np.linalg.inv(self.intrinsics), self.image2_points)
        result = reconstruction.triangulate_points_opencv_2(self.image1_points, self.image2_points, self.intrinsics, self.T_1_to_2,  im1_poses, im2_poses,  self.distortion)
        print('opencv 2')
        print(f'{result.round()}')
        #result=result.T
        self.assertEqual( round(result[0][0]), self.xyz[0][0])
        self.assertEqual( round(result[0][1]), self.xyz[0][1])
        self.assertEqual( round(result[0][2]), self.xyz[0][2])

    '''
    def test_triangulation_prince(self):

        im1_poses = rigid_body_parameters_to_matrix(np.concatenate((self.rvec_1,self.tvec_1)))
        im2_poses = rigid_body_parameters_to_matrix(np.concatenate((self.rvec_2,self.tvec_2)))
        result, _ = reconstruction.get_xyz_method_prince(self.intrinsics, self.image1_points, im1_poses, self.image1_points, im2_poses, image_1=None)
        print('prince')
        print(f'{result}')
        
        self.assertEqual( (result[0]), self.xyz[0][0])
        #self.assertEqual( (result[1]), self.xyz[0][1])
        #self.assertEqual( (result[2]), self.xyz[0][2])
    
    '''

if __name__=='__main__':
    unittest.main()