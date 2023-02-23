import unittest
import reconstruction
import numpy as np
import cv2


class TestReconstruction(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.xyz = np.array([[0,300,0]], dtype='float32')
        rvec_1 = tvec_1 = np.zeros(3)

        self.intrinsics = np.loadtxt(f'/Users/aure/Documents/CARES/code/matching/SuperGlue/calibration/mac_calibration/intrinsics.txt')
        self.distortion = np.loadtxt(f'/Users/aure/Documents/CARES/code/matching/SuperGlue/calibration/mac_calibration/distortion.txt')
        
        self.image1_points, _ = cv2.projectPoints(self.xyz, rvec_1, tvec_1, self.intrinsics, self.distortion)
        
        # apply translation for second image
        y_translation = 10
        self.T_1_to_2 = np.array([
            [1,0,0,0],
            [0,1,0,y_translation],
            [0,0,1,0],
            [0,0,0,1]
        ])

        rvec_2 = tvec_2 = np.zeros(3)
        tvec_2[1]=y_translation

        self.image2_points, _ = cv2.projectPoints(self.xyz, rvec_2, tvec_2, self.intrinsics, self.distortion)
        

        return

    def tearDown(self):
        pass


    def test_triangulation_opencv_2(self):

        result = reconstruction.triangulate_points_opencv_2(self.image1_points, self.image2_points, self.intrinsics, self.T_1_to_2)
        print(f'{result.T.round()}')
        print(f'{self.xyz.round()}')
        result=result.T
        self.assertEqual( round(result[0][0]), self.xyz[0][0])
        self.assertEqual( round(result[0][1]), self.xyz[0][1])
        self.assertEqual( round(result[0][2]), self.xyz[0][2])

    
    

if __name__=='__main__':
    unittest.main()