

# -*- coding: utf-8 -*-

import cv2
import numpy as np
from reconstruction import reconstruct_pairs
from reconstruction_utils.utils import extrinsic_vecs_to_matrix, extrinsic_matrix_to_vecs
from pathlib import Path

from .common_tests import arrays_equal


def test_reconstruction(intrinsics, distortion, xyz):
    """
    testing 3D reconstruction working
    We create fake data by projecting a set of defined 3D points into
    2 different camera views- one with extrinsics as all 0 and the other translated in the X dimension
    """

    # Create reference points. This is in 3D, in camera space. i.e. along z-axis.
    #xyz = np.array([[0.0, 0.0, 200.0],
    #                             [0,0,100]])

    # Project points to first camera
    projected_points_1, _ = cv2.projectPoints(xyz,
                                              rvec=np.zeros((1, 3)), # No rotations or translations, so world coords==camera coords.
                                              tvec=np.zeros((1, 3)),
                                              cameraMatrix=intrinsics,
                                              distCoeffs=distortion)

    # defining extrinsics of second camera. Just translation in x.
    extrinsics_matrix = np.eye(4, 4)
    extrinsics_matrix[0][3] = 10
    rvec_1, tvec_1 = extrinsic_matrix_to_vecs(extrinsics_matrix)
    # RT 4x4 matrix between cam1 and cam 2
    RT = extrinsic_vecs_to_matrix(rvec_1, tvec_1)

    # Project to 2nd camera.
    projected_points_2, _ = cv2.projectPoints(xyz,
                                              rvec=rvec_1,
                                              tvec=tvec_1,
                                              cameraMatrix=intrinsics,
                                              distCoeffs=distortion)

    # undistorting 2d points
    kp1_matched = cv2.undistortPoints(projected_points_1.astype('float32'), intrinsics, distortion, None,
                                      intrinsics)
    kp2_matched = cv2.undistortPoints(projected_points_2.astype('float32'), intrinsics, distortion, None,
                                      intrinsics)

    kp1_matched = kp1_matched.squeeze(1).T  # (2XN)
    kp2_matched = kp2_matched.squeeze(1).T  # (2XN)

    # reconstruct points
    D3_points, colour_mask = reconstruct_pairs('opencv', kp1_matched, kp2_matched, intrinsics, RT)

    # Check reconstructed points equals original point
    print('orig')
    print(xyz)
    print('triang')
    print(D3_points.round())
    arrays_equal(xyz, D3_points)

