# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sksurgerycalibration.video.video_calibration_utils as vu


def test_opencv_triangulate():
    """
    Test to ensure we can project and triangulate using OpenCV.
    """
    intrinsics = np.loadtxt(f'calibration/mac_calibration/intrinsics.txt')
    print(f"\nIntrinsics shape:{intrinsics.shape}")

    distortion = np.loadtxt(f'calibration/mac_calibration/distortion.txt')
    print(f"Distortion shape:{distortion.shape}")

    # Create reference points. This is in 3D, in camera space. i.e. along z-axis.
    points_on_z_axis = np.array([[0.0, 0.0, 200.0]])

    # Project to first camera. Notice that rvec, and tvec are zero, so it's the identity matrix.
    projected_points_1, _ = cv2.projectPoints(points_on_z_axis,
                                              rvec=np.zeros((1, 3)), # No rotations or translations, so world coords==camera coords.
                                              tvec=np.zeros((1, 3)),
                                              cameraMatrix=intrinsics,
                                              distCoeffs=distortion)

    # Note, that with 1 point, the shape of projected_points is [1, 1, 2]. So, its 1 rows, of a 1x2 row vector.
    # For the sake of unit testing, points on z-axis should project to centre of projection, given by intrinsic matrix principle point.
    assert np.isclose(projected_points_1[0][0][0], intrinsics[0][2])
    assert np.isclose(projected_points_1[0][0][1], intrinsics[1][2])

    # So, lets create a second camera. No rotation. Just translate in x.
    extrinsics_matrix = np.eye(4, 4)
    extrinsics_matrix[0][3] = 10
    rvec_1, tvec_1 = vu.extrinsic_matrix_to_vecs(extrinsics_matrix)

    # Project to 2nd camera.
    projected_points_2, _ = cv2.projectPoints(points_on_z_axis,
                                              rvec=rvec_1,
                                              tvec=tvec_1,
                                              cameraMatrix=intrinsics,
                                              distCoeffs=distortion)

    assert not (np.isclose(projected_points_2[0][0][0], intrinsics[0][2])) # As the x-position SHOULD have changed.
    assert np.isclose(projected_points_2[0][0][1], intrinsics[1][2])       # As the y-position should NOT have changed.

    # Firstly: Triangulation in OpenCV only works on undistorted points.
    # Secondly: There are 2 ways to sort out the intrinsics.
    #
    # Option A: Do NOT pass intrinsics as the 5th argument to undistort points.
    #           The undistorted points are then normalised.
    #           The triangulation process then HAS to use the identity matrix as the intrinsics.
    #
    #           undistorted_0 = cv2.undistortPoints(projected_points, intrinsics, distortion) # so, undistorted points normalised.
    #           p_0 = identity @ ext_0                                                        # so, DONT use intrinsics
    #           p_1 = identity @ ext_0                                                        # so, DONT use intrinsics
    #
    # Option B: DO pass intrinsics as the 5th argument to undistort points.
    #           The undistorted points are then in pixel coordiates.
    #           The triangulation process then MUST use the intrinsics matrix.
    #
    #           undistorted_0 = cv2.undistortPoints(projected_points, intrinsics, distortion, None, intrinsics) # so, undistorted points in pixels.
    #           p_0 = intrinsics @ ext_0                                                                        # so, DO use intrinsics
    #           p_1 = intrinsics @ ext_0                                                                        # so, DO use intrinsics
    #
    # Both options are equivalent. You just must be consistent, and pick one. Let's pick option B.

    projected_points_1_undistorted = cv2.undistortPoints(projected_points_1, intrinsics, distortion, None, intrinsics)
    projected_points_2_undistorted = cv2.undistortPoints(projected_points_2, intrinsics, distortion, None, intrinsics)

    projected_points_1_undistorted_squeezed = projected_points_1_undistorted.squeeze(1)
    projected_points_2_undistorted_squeezed = projected_points_2_undistorted.squeeze(1)

    print(f"projected_points_1_undistorted_squeezed is:\n{projected_points_1_undistorted_squeezed}")
    print(f"projected_points_2_undistorted_squeezed is:\n{projected_points_2_undistorted_squeezed}")

    # First camera, we say has a world_to_camera matrix of identity.
    r_0 = np.eye(3, 3)
    ext_0 = np.zeros((3, 4))
    ext_0[:3, :3] = r_0
    p_0 = intrinsics @ ext_0
    print(f"p_0 is:\n{p_0}")

    # Second camera, has a different world_to_camera matrix.
    ext_tmp = vu.extrinsic_vecs_to_matrix(rvec_1, tvec_1)
    ext_1 = np.zeros((3, 4))
    ext_1[:3, :3] = ext_tmp[:3, :3]
    ext_1[:3, 3] = ext_tmp[:3, 3]
    p_1 = intrinsics @ ext_1
    print(f"p_1 is:\n{p_1}")

    homogeneous_triangulated_points = cv2.triangulatePoints(projMatr1=p_0,
                                                            projMatr2=p_1,
                                                            projPoints1=np.transpose(projected_points_1_undistorted_squeezed),
                                                            projPoints2=np.transpose(projected_points_2_undistorted_squeezed))
    triangulated_points = cv2.convertPointsFromHomogeneous(np.transpose(homogeneous_triangulated_points))

    # Moment of truth.
    print(triangulated_points)

    # Check that this equals the point we put in: points_on_z_axis = np.array([[0.0, 0.0, 200.0]])
    assert np.isclose(triangulated_points[0][0][0], points_on_z_axis[0][0])
    assert np.isclose(triangulated_points[0][0][1], points_on_z_axis[0][1])
    assert np.isclose(triangulated_points[0][0][2], points_on_z_axis[0][2])



if __name__=='__main__':
    test_opencv_triangulate()