import unittest
import reconstruction
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as spr
#import reconstruction_utils.utils
from reconstruction_utils import utils

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
        self.xy = np.array([
            [0,10],
            [2,10],
            [12,13],
            [10,79],
            [10,30],
            ], dtype='float32')
        
        print('original xy')
        print(f'{self.xy.round()}')

        self.intrinsics = np.loadtxt(f'/Users/aure/Documents/CARES/code/matching/SuperGlue/calibration/mac_calibration/intrinsics.txt')
        self.distortion = np.loadtxt(f'/Users/aure/Documents/CARES/code/matching/SuperGlue/calibration/mac_calibration/distortion.txt')
        
        return

    def tearDown(self):
        pass
    


    def test_normalise(self):
        normalised_xy = utils.normalise(self.xy, self.intrinsics)
        un_normalised = utils.un_normalise(normalised_xy, self.intrinsics)
        
        print('normalised')
        print(normalised_xy)
        print('un normalised 2')
        print(f'{un_normalised.round()}')
        


    
    

if __name__=='__main__':
    unittest.main()