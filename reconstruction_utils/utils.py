import numpy as np
import matplotlib.pyplot as plt
import cv2
import sksurgerycore.transforms.matrix as stm
import copy


def manually_match_features(img1, img2):
    ''' Function used to manually annotate feature matches between images, returning the annotated matches.

    The function will plot the two input images side by side.
    The user then selects matching points between images, marked on the images as the user does so.
    When the user closes the window of the subplots, the matched keypoints are returned.

    Args:
        img1 (UxV):
        img2 (UxV):

    Returns:

    '''

    # Define a callback function for mouse clicks
    matches = []

    count_right = 0
    count_left = 0

    def onclick(event):
        nonlocal count_left, count_right
        # Check which axes was clicked
        if event.inaxes == ax[0]:
            print('point 1')
            matches.append([event.xdata, event.ydata, None, None])
            ax[0].scatter(event.xdata, event.ydata, marker='+', s=100)
            ax[0].text(event.xdata, event.ydata, count_left)
            plt.draw()
            count_left += 1
        elif event.inaxes == ax[1]:
            print('point 2')
            matches[-1][2:] = [event.xdata, event.ydata]
            ax[1].scatter(event.xdata, event.ydata, marker='+', s=100)
            ax[1].text(event.xdata, event.ydata, count_right)
            plt.draw()
            count_right += 1

    # Wait for the user to close the plot
    # Display the images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img1)
    ax[1].imshow(img2)

    # Attach the callback function to the figure
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Convert the matches to numpy arrays
    matches = np.array(matches)

    return matches[:, :2], matches[:, 2:]


def multiply_points_by_transform(D3_points, T):
    """
    Applies a 4x4 transformation matrix to a set of 3D points.

    Args:
        D3_points (numpy.ndarray): Array of 3D points with shape (N, 3).
        T (numpy.ndarray): 4x4 transformation matrix.

    Returns:
        (numpy.ndarray) Array of transformed 3D points with shape (N, 3).
    """
    D3_hom = cv2.convertPointsToHomogeneous(D3_points).squeeze()
    D3_transformed_points_hom = T @ D3_hom.T
    D3_transformed = cv2.convertPointsFromHomogeneous(D3_transformed_points_hom.T).squeeze()
    return D3_transformed


def estimate_camera_poses(kp1_matched, kp2_matched, K, recover_pose_method='opencv', essential_mat_method='opencv'):
    """ Computes the relative pose between two images given their keypoints and camera intrinsics.

    Args:
        kp1_matched (numpy.ndarray), (2, N): Array of keypoints in the first image
        kp2_matched (numpy.ndarray), (2, N): Array of keypoints in the second image
        K (numpy.ndarray), (3,3): Camera intrinsic matrix.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the rotation matrix and translation vector
        that transforms points in the first camera frame to the second.
    """

    #kp1_norm = cv2.undistortPoints(np.expand_dims(kp1_matched, axis=2), K, None)
    #kp2_norm = cv2.undistortPoints(np.expand_dims(kp2_matched, axis=2), K, None)
    kp1_matched = kp1_matched.T
    kp2_matched = kp2_matched.T

    # Estimate the essential matrix using the normalized points
    if essential_mat_method == 'opencv':
        # method 1: using opencv
        E, _ = cv2.findEssentialMat(kp1_matched, kp1_matched, K)
    else:
        # method 2: from fundamental matrix
        F, _ = cv2.findFundamentalMat(kp1_matched.T, kp2_matched.T, cv2.FM_RANSAC)
        E = K.T @ F @ K

    # recover pose from essential matrix
    if recover_pose_method=='opencv':
        # method 1) opencv
        _, R, t, _ = cv2.recoverPose(E, kp1_matched, kp2_matched, K)
    else:
        # method 2) svd decomposition
        U, _, Vt = np.linalg.svd(E)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R = np.matmul(np.matmul(U, W), Vt)
        t = U[:, 2]

        # Check for the correct rotation matrix
        if np.linalg.det(R) < 0:
            R = -R
            t = -t

    return R, t


def extrinsic_vecs_to_matrix(rvec, tvec):
    """
    Method to convert rvec and tvec to a 4x4 matrix.
    :param rvec: [3x1] ndarray, Rodrigues rotation params
    :param rvec: [3x1] ndarray, translation params
    :return: [3x3] ndarray, Rotation Matrix
    """
    rotation_matrix = (cv2.Rodrigues(rvec))[0]
    transformation_matrix = \
        stm.construct_rigid_transformation(rotation_matrix, tvec)
    return transformation_matrix


def extrinsic_matrix_to_vecs(transformation_matrix):
    """
    Method to convert a [4x4] rigid body matrix to an rvec and tvec.
    :param transformation_matrix: [4x4] rigid body matrix.
    :return [3x1] Rodrigues rotation vec, [3x1] translation vec
    """
    rmat = transformation_matrix[0:3, 0:3]
    rvec = (cv2.Rodrigues(rmat))[0]
    tvec = np.ones((3, 1))
    tvec[0:3, 0] = transformation_matrix[0:3, 3]
    return rvec, tvec


def get_projection_matrices(rvec_1, rvec_2, tvec_1, tvec_2, K):
    RT0 = extrinsic_vecs_to_matrix(rvec_1, tvec_1)
    P0 = K @ RT0[:3]  # Projection matrix

    RT1 = extrinsic_vecs_to_matrix(rvec_2, tvec_2)
    P1 = K @ RT1[:3]
    return P0, P1


def img_poses_reformat(im_poses):
    """
    function extractin pose information of 4x4 matrix to be used for prince reconstruction method

    Args:
        im_poses (4x4): np array 4x4 pose matrix

    Returns:
        decomposed pose matrix to be used for prince method reconstruction
    """
    # MATRICES OF EQUATION 1
    tx = im_poses[0, -1]
    ty = im_poses[1, -1]
    tz = im_poses[2, -1]

    w31 = im_poses[2, 0]
    w11 = im_poses[0, 0]
    w21 = im_poses[1, 0]

    w32 = im_poses[2, 1]
    w12 = im_poses[0, 1]
    w22 = im_poses[1, 1]

    w33 = im_poses[2, 2]
    w13 = im_poses[0, 2]
    w23 = im_poses[1, 2]

    return tx, ty, tz, w31, w11, w21, w32, w12, w22, w33, w13, w23


def get_matched_keypoints_superglue(pair_match):
    '''
    function to get keypoints from superglue
    input:
        - pair_match: npz file of image pairs
    output:
        - kp1_matched, kp2_matched: np arrays of marched points
    '''
    # MATCHES info of the two images
    npz = np.load(pair_match)

    kp1 = npz['keypoints0']  # keypoints in first image
    kp2 = npz['keypoints1']  # keypoints in second image
    matches = npz['matches']  # matches- for each point in kp1, finds match in kp2. If -1-> no match

    # selecting in order the indeces of the matching points
    kp1_matched = kp1[matches > -1]  # selecting all indeces that are matches in im1
    kp2_matched = kp2[matches[matches > -1]]  # selecting points whose indeces are matches in sim2
    return kp1_matched, kp2_matched


def get_matched_keypoints_sift(img1_original, img2_original):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # orb = cv2.ORB_create()

    # convert 2 images to gray
    img1 = cv2.cvtColor(img1_original, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_original, cv2.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # find the keypoints and descriptors with ORB
    # kp1, des1 = orb.detectAndCompute(img1,None)
    # kp2, des2 = orb.detectAndCompute(img2,None)

    # feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    '''
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    '''
    # Apply ratio test
    good_kp1_matched = []
    good_kp2_matched = []
    good = []

    for m, n in matches:

        if m.distance < 0.8 * n.distance:
            good.append(m)
            # https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python
            # extracting indexes of matched idx from images
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx
            good_kp1_matched.append(kp1[img1_idx].pt)
            good_kp2_matched.append(kp2[img2_idx].pt)
    '''
    for m in matches:
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
        good_kp1_matched.append(kp1[img1_idx].pt)
        good_kp2_matched.append(kp2[img2_idx].pt)
    '''
    # converting to np array
    kp1_matched = np.asarray(good_kp1_matched)  # selecting all indeces that are matches in im1
    kp2_matched = np.asarray(good_kp2_matched)  # selecting points whose indeces are matches in sim2

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, img2, flags=2)
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.savefig('plot.png')
    return kp1_matched, kp2_matched
