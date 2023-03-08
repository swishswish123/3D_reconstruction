import unittest
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as spr
from reconstruction_utils import utils
from reconstruction_utils.utils import rigid_body_parameters_to_matrix
from common_tests import test_arrays_equal


class TestReconstruction(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # creating image points example
        num_points = 10 # how many random points are created to simulate image points
        project_path = Path(__file__).parent.resolve()

        # loading intrinsics and distortion
        self.intrinsics = np.loadtxt(f'{project_path}/../calibration/mac_calibration/intrinsics.txt')
        self.distortion = np.loadtxt(f'{project_path}/../calibration/mac_calibration/distortion.txt')
        
        # generating xy random points (image)
        self.xy  = np.random.uniform(low=0, high=self.intrinsics[1,2], size=(num_points,2))
        
        print('original xy')
        print(f'{self.xy.round()}')

        return

    def tearDown(self):
        pass
    

    def test_normalise(self):
        '''
        this function tests if normalising the image works (inverse of intrinsics, then uninvert)
        '''
        normalised_xy = utils.normalise(self.xy, self.intrinsics)
        un_normalised = utils.un_normalise(normalised_xy, self.intrinsics)

        # testing the two arrays are the same
        ground_truth = self.xy
        acual = un_normalised

        test_arrays_equal(ground_truth, acual)
        
    def test_rigid_body_parameters_to_matrix(self):
        # defining extrinsics of two camera views
        rvec_1 = np.zeros(3)
        tvec_1 = np.zeros(3)

        rvec_2 = np.zeros(3)
        tvec_2 = np.zeros(3) # x,y,z translation of second img
        tvec_2[0] = 110   #x
        tvec_2[1] = 100   #y
        tvec_2[2] = 200   #z

        # transform between two camera views
        mat1 = rigid_body_parameters_to_matrix(np.concatenate([rvec_1, tvec_1])) 
        mat2 = rigid_body_parameters_to_matrix(np.concatenate([rvec_2, tvec_2])) 
        T_1_to_2 = mat2 @ np.linalg.inv(mat1) 

        T_1_to_2_ground_truth = np.array([
            [1,0,0,tvec_2[0]],
            [0,1,0,tvec_2[1]],
            [0,0,1,tvec_2[2]],
            [0,0,0,1]
        ])

        test_arrays_equal(T_1_to_2, T_1_to_2_ground_truth)
        

if __name__=='__main__':
    unittest.main()