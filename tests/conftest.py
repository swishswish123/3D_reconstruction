import numpy as np
import pytest
from pathlib import Path
import cv2
from reconstruction_utils.utils import get_projection_matrices

'''
@pytest.fixture
def test_arrays_equal(ground_truth, acual):
    assert len(ground_truth) == len(acual)

    for normalised_row, un_normalised_row in zip(ground_truth, acual):
        for normalised_item, un_normalised_item in zip(normalised_row, un_normalised_row):
            assert np.isclose(round(normalised_item) ,  round(un_normalised_item))
'''


##### CAMERA PARAMS
@pytest.fixture
def project_path():
    return Path(__file__).parent.resolve()


@pytest.fixture
def intrinsics(project_path):
    return np.loadtxt(f'{project_path}/../calibration/mac_calibration/intrinsics.txt')


@pytest.fixture
def distortion(project_path):
    return np.loadtxt(f'{project_path}/../calibration/mac_calibration/distortion.txt')


# ####### RANDOM POINTS
@pytest.fixture
def xyz(intrinsics):
    random_xyz = np.random.randint(low=0, high=[50,100,100], size=(5,3)).astype('float64')
    xyz = np.array([
        [1,2,3],
        [10,20,30],
        [100,200,300],
        [12,25,80],
        [17, 7, 800],
        [10, 57, 80],
    ], dtype='float64')

    return xyz


@pytest.fixture
def xyz_hom(xyz):
    return cv2.convertPointsToHomogeneous(xyz).squeeze()


# ####### CAMERA POSES AND PROJECTION MATRICES
@pytest.fixture
def get_cam_poses():
    rvec_1 = np.zeros(3)
    tvec_1 = np.zeros(3)
    
    # rotation -45 degrees to 45 degrees
    rvec_2 = np.random.uniform(low=-45, high=45, size=(3))
    # translation -500mm to 500mm
    tvec_2 = np.random.uniform(low=-500, high=500, size=(3))
    return (rvec_1, tvec_1, rvec_2, tvec_2)


@pytest.fixture
def projection_matrices(get_cam_poses, intrinsics):
    rvec_1, rvec_2, tvec_1, tvec_2 = get_cam_poses
    P1, P2 = get_projection_matrices(rvec_1, rvec_2, tvec_1, tvec_2, intrinsics)
    return P1, P2


@pytest.fixture
def rvec_2():
    return


@pytest.fixture
def tvec_2():
    return 