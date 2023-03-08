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
        
        print('normalised')
        print(normalised_xy)
        print('un normalised 2')
        print(f'{un_normalised.round()}')

        # testing the two arrays are the same
        ground_truth = self.xy
        acual = un_normalised

        test_arrays_equal(ground_truth, acual)
        

if __name__=='__main__':
    unittest.main()