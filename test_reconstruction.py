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
                              [56.0, 0.8, 21.0],
                              [5.0, 8, 20],
                              [1,2,3]
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
        print('TESTING OPENCV AND PROJECTION IS THE SAME')
        # testing if projection using opencv method and proj matrices same
        # OPENCV
        image1_points, _ = cv2.projectPoints(self.xyz, self.rvec_1, self.tvec_1, self.intrinsics, self.distortion)
        image1_points = image1_points.squeeze()      

        image2_points, _ = cv2.projectPoints(self.xyz, self.rvec_2, self.tvec_2, self.intrinsics, self.distortion)
        image2_points = image2_points.squeeze()

        # Projection wa
        P1, P2 = get_projection_matrices(self.rvec_1,self.rvec_2, self.tvec_1, self.tvec_2, self.intrinsics)
        points = self.xyz_hom.T
        im1 = P1@points
        im1 = im1[:2, :] / im1[2, :]

        im2 = P2@points
        im2 = im1[:2, :] / im2[2, :]

        print('image 1 points')
        print(image1_points)
        print(im1.T)
        print('image 2 points')
        print(image2_points)
        print(im2.T)

        plt.figure()
        plt.scatter(image1_points[:,0], image1_points[:,0])
        plt.scatter(im1[0], im1[1])
        plt.savefig("mygraph.png")


    def test_projection_loop(self):
        P0, P1 = get_projection_matrices(self.rvec_1,self.rvec_2, self.tvec_1, self.tvec_2, self.intrinsics)
        print(' loop')

        im0_points_all = np.zeros((self.xyz_hom.shape[0],2))
        for idx, point in enumerate(self.xyz_hom):
            # Projecting 3D point to 2D using projection matrices
            im_point_0 = P0 @ point # project this point using the first camera pose
            im0_points_all[idx,:] = im_point_0[:2]/im_point_0[-1] # normalize as we are in homogenuous coordinates
            
        print(im0_points_all)  
        # Projection wa
        print('no loop')
        points = self.xyz_hom.T
        im1 = P0@points
        im1 = (im1[:2, :] / im1[2, :]).T
        print(im1)

        self.assertEqual(im0_points_all.all(), im1.all())



    def test_triangulation(self):
        print('TESTING BASIC TRIANGULATION')
        P0, P1 = get_projection_matrices(self.rvec_1,self.rvec_2, self.tvec_1, self.tvec_2, self.intrinsics)
        points = self.xyz_hom.T
        
        # projecting points
        im0 = P0@points
        im0 = im0[:2, :] / im0[2, :]

        im1 = P1@points
        im1 = im1[:2, :] / im1[2, :]

        res_1 = cv2.triangulatePoints(P0, P1, im0, im1) 
        res_1 = res_1[:3] / res_1[3, :]
        print('reconstruction 1')
        print(res_1.T)

        output_points_all = np.zeros((im1.shape[1],3))
        for i,point in enumerate(self.xyz_hom):
            # Projecting 3D point to 2D using projection matrices
            im_point_0 = P0 @ point # project this point using the first camera pose
            im_point_0 = im_point_0/im_point_0[-1] # normalize as we are in homogenuous coordinates

            im_point_1 = P1 @ point
            im_point_1 = im_point_1/im_point_1[-1] # normalize as we are in homogenuous coordinates
           
            res = cv2.triangulatePoints(P0, P1, im_point_0[:2], im_point_1[:2]) 
            output_points_all[i,:] = (res[:3]/res[-1]).squeeze()
        
        print('RECON2')
        print(output_points_all)


    def test_triangulation_opencv_2_projection(self):
        print('TESTING FUNCTION with projection pnts')
        print('original')
        print(self.xyz_hom.round())
        P0, P1 = get_projection_matrices(self.rvec_1,self.rvec_2, self.tvec_1, self.tvec_2, self.intrinsics)
        points = self.xyz_hom.T
        
        # projecting points
        im0 = P0@points
        image1_points = im0[:2, :] / im0[2, :]

        im1 = P1@points
        image2_points = im1[:2, :] / im1[2, :]


        im1_poses = rigid_body_parameters_to_matrix(np.concatenate((self.rvec_1,self.tvec_1)))
        im2_poses = rigid_body_parameters_to_matrix(np.concatenate((self.rvec_2,self.tvec_2)))

        #im1_norm = np.matmul(np.linalg.inv(self.intrinsics), self.image1_points)
        #im2_norm = np.matmul(np.linalg.inv(self.intrinsics), self.image2_points)
        #result = reconstruction.triangulate_points_opencv_2(image1_points, image2_points, self.intrinsics, self.T_1_to_2,  im1_poses, im2_poses,  self.distortion)
        result = reconstruction.triangulate_points_opencv_2(image1_points, image2_points, self.intrinsics,self.rvec_1,self.rvec_2, self.tvec_1, self.tvec_2, T_1_to_2=self.T_1_to_2, poses1=im1_poses, poses2=im2_poses,  distortion=self.distortion)
        result = result.T
        print('recon ')
        print(f'{result.round()}')
        #result=result.T
        #self.assertEqual( round(result[0][0]), self.xyz[0][0])
        #self.assertEqual( round(result[0][1]), self.xyz[0][1])
        for row, point_x in enumerate(self.xyz):
            print(point_x)
            for col, original in enumerate(point_x):
                self.assertEqual( round(result[row,col]), round(original))
                self.assertEqual( round(result[row,col]), round(original))
                self.assertEqual( round(result[row,col]), round(original))

    '''
    def test_triangulation_opencv_2_opencvprojection(self):
        print('TESTING opencv recon')

        im1_poses = rigid_body_parameters_to_matrix(np.concatenate((self.rvec_1,self.tvec_1)))
        im2_poses = rigid_body_parameters_to_matrix(np.concatenate((self.rvec_2,self.tvec_2)))

        image1_points, _ = cv2.projectPoints(self.xyz, self.rvec_1, self.tvec_1, self.intrinsics, self.distortion)
        image1_points = image1_points.squeeze()      

        image2_points, _ = cv2.projectPoints(self.xyz, self.rvec_2, self.tvec_2, self.intrinsics, self.distortion)
        image2_points = image2_points.squeeze()


        #im1_norm = np.matmul(np.linalg.inv(self.intrinsics), self.image1_points)
        #im2_norm = np.matmul(np.linalg.inv(self.intrinsics), self.image2_points)
        result = reconstruction.triangulate_points_opencv_2(image1_points, image2_points, self.intrinsics,self.rvec_1,self.rvec_2, self.tvec_1, self.tvec_2, T_1_to_2=self.T_1_to_2, poses1=im1_poses, poses2=im2_poses,  distortion=self.distortion)
        print('opencv 2')
        print(f'{result.round()}')
        #result=result.T
        self.assertEqual( round(result[0][0]), self.xyz[0][0])
        self.assertEqual( round(result[0][1]), self.xyz[0][1])
        self.assertEqual( round(result[0][2]), self.xyz[0][2])
    '''
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