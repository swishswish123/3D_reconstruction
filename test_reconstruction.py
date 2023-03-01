import unittest
import reconstruction
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as spr
import math
import matplotlib.pyplot as plt
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

class TestReconstruction(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.xyz = np.array([
            [95,10,44],
            [550,0.5,10.5],
            [12,13,43],
            ], dtype='float32')

        self.xyz  = np.array([[940, 530, 50],
                              [5.0, 10., 20.0],
                              [500.0, 110., 20.0],
                              [56.0, 0.8, 21.0]
                                
                                ]) # define an arbitrary 3D point in world coordinates
        self.xyz_hom = cv2.convertPointsToHomogeneous(self.xyz).squeeze()

        print('original xyz')
        print(f'{self.xyz_hom.round()}')
        
        
        
        self.rvec_1 = np.zeros(3)
        self.tvec_1 = np.zeros(3)

        self.intrinsics = np.loadtxt(f'/Users/aure/Documents/CARES/code/matching/SuperGlue/calibration/mac_calibration/intrinsics.txt')
        self.distortion = np.loadtxt(f'/Users/aure/Documents/CARES/code/matching/SuperGlue/calibration/mac_calibration/distortion.txt')
        
        # apply translation for second image
        y_translation = 10

        self.T_1_to_2 = np.array([
            [1,0,0,0],
            [0,1,0,y_translation],
            [0,0,1,0],
            [0,0,0,1]
        ])

       
        self.image1_points, _ = cv2.projectPoints(self.xyz, self.rvec_1, self.tvec_1, self.intrinsics, self.distortion)
        self.image1_points = self.image1_points.squeeze()
        #print('original projected points')
        #print(self.image1_points)        

        self.rvec_2 = np.zeros(3)
        self.tvec_2 = np.zeros(3)
        self.tvec_2[1]=-y_translation

        self.image2_points, _ = cv2.projectPoints(self.xyz, self.rvec_2, self.tvec_2, self.intrinsics, self.distortion)
        self.image2_points = self.image2_points.squeeze()

        return

    def tearDown(self):
        pass
    
    '''
    def test_gpt(self):
        
        result = reconstruction.reconstruction_gpt(self.image1_points, self.image2_points, self.intrinsics, self.T_1_to_2)
        print('gpt')
        print(result)
    '''

    def test_projection(self):
        #points = np.array([[2.0, 0.5, 15., 1.], 
        #[3.0, 0.5, 15., 1.]
        #])

        #hom_xyz = cv2.convertPointsToHomogeneous(self.xyz).squeeze().T
        R = eulerAnglesToRotationMatrix([0,0,0]) # Rotation matrix:
        K = self.intrinsics

        T0 = np.array([0,0,0]) # Translation vector
        RT0 = np.zeros((3,4))  # combined Rotation/Translation matrix
        RT0[:3,:3] = R
        RT0[:3, 3] = T0
        P0 = K@RT0 # Projection matrix

        T1 = np.array([10,10,0.])
        RT1 = np.zeros((3,4))
        RT1[:3,:3] = R
        RT1[:3, 3] = -T1
        P1 = K@ RT1


        '''
        extrinsics_0 = np.zeros((3,4))  # combined Rotation/Translation matrix
        extrinsics_0[:3,:3] = R
        extrinsics_0[:3, 3] = T0

        extrinsics_1 = np.zeros((3,4))  # combined Rotation/Translation matrix
        extrinsics_1[:3,:3] = R
        extrinsics_1[:3, 3] = -T1
        '''
        #extrinsics = np.hstack((np.identity(3), np.zeros((3, 1))))

        #P0 = self.intrinsics@ RT0 # Projection matrix
        #P1 = self.intrinsics@ RT1 # Projection matrix

        
        for point in self.xyz_hom:

            #P1 = self.intrinsics @ extrinsics
            #P2 = self.intrinsics @ self.T_1_to_2[:3]
            icsp0 = P0 @ point # project this point using the first camera pose
            icsp0 = icsp0/icsp0[-1] # normalize as we are in homogenuous coordinates

            icsp1 = P1 @ point
            icsp1 = icsp1/icsp1[-1] # normalize as we are in homogenuous coordinates
            #icsp0 = icsp0[:2]/icsp0[-1] # normalize as we are in homogenuous coordinates
            #icsp0 = cv2.convertPointsFromHomogeneous(icsp0.T)
            #icsp0 = cv2.undistortPoints(icsp0, self.intrinsics, self.distortion)
        

            #icsp1 = cv2.convertPointsFromHomogeneous(icsp1.T)
            #icsp1 = cv2.undistortPoints(icsp1, self.intrinsics, self.distortion)
            #icsp1[:2]/icsp1[-1]

            #projected_0 = hom_xyz@extrinsics@self.intrinsics # project this point using the first camera pose
            #projected_0 = self.intrinsics@(P0@hom_xyz)
            #projected_0 = projected_0/projected_0[-1] # normalize as we are in homogenuous coordinates    
            '''
            print('opencv points')
            print(self.image1_points)
            print('projected points P1')
            print(icsp0)

            print('opencv points')
            print(self.image2_points)
            print('projected points P1')
            print(icsp1)
            '''
            #print(icsp0[:2])
            res = cv2.triangulatePoints(P0, P1, icsp0[:2], icsp1[:2]) 
            #print(res)
            res = res[:3]/res[-1]
            #print(res)
            

            plt.figure()
            plt.scatter(self.image1_points[:,0], self.image1_points[:,0])
            plt.scatter(icsp1[0], icsp1[1])
            plt.savefig("mygraph.png")

    
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