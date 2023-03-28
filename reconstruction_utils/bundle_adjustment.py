"""
file containing all functions for performing bundle adjustment.
All info taken by following: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html#
"""

from scipy.optimize import least_squares
import time
import numpy as np
from scipy.sparse import lil_matrix


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indeces, point_indeces, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    n_cameras: number of cameras or views (in my case num of frames)
    n_points: number of 3d points
    camera_indices: index of which camera was used to triangulate point-  Each element of camera_indices is an integer representing the index of the camera that generated the corresponding 2D point.
    point_indeces: : Each element of point_indices is an integer representing the index of the 3D point that generated the corresponding 2D point.

    """
    # going back to same shape as before camera params:
    # [rx,ry,rz,,tx,ty,tz,focal_distance, dist1, dist2]
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    # recovering 3d poinys to same as before
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indeces], camera_params[camera_indeces])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indeces, point_indeces):
    m = camera_indeces.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indeces.size)
    for s in range(9):
        A[2 * i, camera_indeces * 9 + s] = 1
        A[2 * i + 1, camera_indeces * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indeces * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indeces * 3 + s] = 1

    return A


def perform_bundle_adjustment(camera_params, camera_indeces, point_indeces, points_2d, points_3d):
    """
    Performs bundle adjustment to refine the camera poses and 3D points based on a set of 2D-3D correspondences.

    Args:
        camera_params (numpy.ndarray): The initial camera parameters, represented as a flattened array of size (n_cameras * 9),
            where n_cameras is the number of cameras. Each camera's parameters are represented as a 9-element vector:
            (rx, ry, rz, tx, ty, tz, f, k1, k2), where (rx, ry, rz) are the rotation angles in radians, (tx, ty, tz) are the
            translation parameters in the camera frame, and (f, k1, k2) are the intrinsic parameters (focal length and radial
            distortion coefficients). The cameras are assumed to be calibrated and the distortion model is assumed to be
            radial-tangential.
        camera_indeces (numpy.ndarray): An array of integers of size (n_observations,), where n_observations is the number of
            2D-3D correspondences. Each element represents the index of the camera (view) that observed the corresponding
            2D point.
        point_indeces (numpy.ndarray): An array of integers of size (n_observations,), where n_observations is the number of
            2D-3D correspondences. Each element represents the index of the 3D point that corresponds to the corresponding
            2D point.
        points_2d (numpy.ndarray): An array of shape (n_observations, 2) containing the 2D coordinates of the observed points.
        points_3d (numpy.ndarray): An array of shape (n_points, 3) containing the initial estimates of the 3D points.

    Returns:
        A tuple (poses, points_3d) containing the refined camera poses and 3D points, respectively. The camera poses are
        represented as a numpy array of shape (n_cameras, 9), where each row corresponds to a camera and each row contains
        the 9 camera parameters (rx, ry, rz, tx, ty, tz, f, k1, k2). The 3D points are represented as a numpy array of shape
        (n_points, 3), where each row contains the 3D coordinates of a point.
    """
    n_cameras = camera_params.shape[0]  # number of cameras or views (frames in my case)
    n_points = points_3d.shape[0]  # number of points triangulated

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    # calculating residuals
    f0 = fun(x0, n_cameras, n_points, camera_indeces, point_indeces, points_2d)

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indeces, point_indeces)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indeces, point_indeces, points_2d))
    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    # obtaining new 3d points and poses after correction
    poses = res.x[:n_cameras * 9].reshape((n_cameras, 9))
    # recovering 3d poinys to same as before
    points_3d = res.x[n_cameras * 9:].reshape((n_points, 3))

    return poses, points_3d

