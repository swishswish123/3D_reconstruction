import unittest
import reconstruction
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as spr

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


class TestReconstruction(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.xyz = np.array([
            [95,10,44],
            [550,0.5,10.5],
            [12,13,43],
            [10,79,76],
            [10,30,50],
            ], dtype='float32')
        
        print('original xyz')
        print(f'{self.xyz.round()}')
        self.rvec_1 = self.tvec_1 = np.zeros(3)

        self.intrinsics = np.loadtxt(f'/Users/aure/Documents/CARES/code/matching/SuperGlue/calibration/mac_calibration/intrinsics.txt')
        self.distortion = np.loadtxt(f'/Users/aure/Documents/CARES/code/matching/SuperGlue/calibration/mac_calibration/distortion.txt')
        
        self.image1_points, _ = cv2.projectPoints(self.xyz, self.rvec_1, self.tvec_1, self.intrinsics, self.distortion)
        
        # apply translation for second image
        y_translation = 10

        self.T_1_to_2 = np.array([
            [1,0,0,0],
            [0,1,0,y_translation],
            [0,0,1,0],
            [0,0,0,1]
        ])

        self.rvec_2 = np.zeros(3)
        self.tvec_2 = np.zeros(3)
        self.tvec_2[1]=y_translation

        self.image2_points, _ = cv2.projectPoints(self.xyz, self.rvec_2, self.tvec_2, self.intrinsics, self.distortion)
        

        return

    def tearDown(self):
        pass

    def test_gpt(self):
        
        result = reconstruction.reconstruction_gpt(self.image1_points, self.image2_points, self.intrinsics, self.T_1_to_2)
        print('gpt')
        print(result)

    def test_triangulation_opencv_2(self):

        result = reconstruction.triangulate_points_opencv_2(self.image1_points, self.image2_points, self.intrinsics, self.T_1_to_2)
        print('opencv 2')
        print(f'{result.T.round()}')
        result=result.T
        self.assertEqual( round(result[0][0]), self.xyz[0][0])
        self.assertEqual( round(result[0][1]), self.xyz[0][1])
        self.assertEqual( round(result[0][2]), self.xyz[0][2])

    
    def test_triangulation_prince(self):

        im1_poses = rigid_body_parameters_to_matrix(np.concatenate((self.rvec_1,self.tvec_1)))
        im2_poses = rigid_body_parameters_to_matrix(np.concatenate((self.rvec_2,self.tvec_2)))
        result, _ = reconstruction.get_xyz_method_prince(self.intrinsics, self.image1_points, im1_poses, self.image1_points, im2_poses, image_1=None)
        print('prince')
        print(f'{result}')
        
        self.assertEqual( (result[0]), self.xyz[0][0])
        #self.assertEqual( (result[1]), self.xyz[0][1])
        #self.assertEqual( (result[2]), self.xyz[0][2])
    
    

if __name__=='__main__':
    unittest.main()