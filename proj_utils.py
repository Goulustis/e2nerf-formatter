import cv2
import numpy as np
import os.path as osp
import scipy.ndimage as ndimage
import scipy.signal as signal
import imageio
from misc_utils import parallel_map

def proj_3d_pnts(img, intrinsics, extrinsics, pnts_3d, pnt_idxs=None, dist_coeffs=None):
    """
    Project a list of 3D points onto an image and label them with their indices.
    
    Inputs:
        img: Image array (np.array [h, w, 3])
        intrinsics: Camera intrinsic matrix (np.array [3, 3])
        extrinsics: Camera extrinsic matrix (np.array [4, 4])
        pnt_idxs: Indices of points to be projected (np.array [n])
        pnts_3d: 3D points in world coordinates (np.array [n, 3])
        dist_coeffs: Distortion coefficients (np.array [4, ]) = k1, k2, p1, p2
    Returns:
        proj_pnts: Projected 2D points (np.array [n, 2])
        img_with_pnts: Image with projected points and labels drawn
    """
    # Extract rotation and translation from the extrinsic matrix
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]

    # Filter points based on indices
    selected_pnts_3d = pnts_3d

    # Project points
    proj_pnts_2d, _ = cv2.projectPoints(selected_pnts_3d, R, T, intrinsics, dist_coeffs)

    # Draw points and labels on the image
    if img is not None:
        img_with_pnts = img.copy()
        for i, p in enumerate(proj_pnts_2d):
            try:
                point = tuple(p[0].astype(int))
                cv2.circle(img_with_pnts, point, 5, (0, 255, 0), -1)
                if pnt_idxs is not None:
                    cv2.putText(img_with_pnts, str(pnt_idxs[i]), (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                print("ERROR:", e)
    else:
        img_with_pnts = img

    return proj_pnts_2d, img_with_pnts

def pnp_extrns(objpnts, pnts_2d, intrxs, dist, ini_R=None, ini_tvec=None):
    pnts_2d = pnts_2d.squeeze()

    if ini_R is None:
        success, rvec, tvec = cv2.solvePnP(objpnts, pnts_2d, intrxs, dist)
    else:
        initial_rvec, _ = cv2.Rodrigues(ini_R)
        # success, rvec, tvec, inliers = cv2.solvePnPRansac(objpnts, pnts_2d, intrxs, dist, flags=cv2.SOLVEPNP_ITERATIVE, rvec=initial_rvec, tvec=ini_tvec)
        success, rvec, tvec = cv2.solvePnP(
                        objpnts, 
                        pnts_2d, 
                        intrxs, 
                        dist, 
                        rvec=initial_rvec, 
                        tvec=ini_tvec, 
                        useExtrinsicGuess=True, 
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
    R, _ = cv2.Rodrigues(rvec)

    assert success
    return R, tvec


def execute_triangulate_points(points1, points2, intrinsics1, dist_coeffs1, intrinsics2, dist_coeffs2, extrinsics1, extrinsics2):
    """
    Triangulate 3D points from two sets of corresponding 2D points.

    :param points1: List of (x, y) tuples of points in the first image.
    :param points2: List of (x, y) tuples of points in the second image.
    :param intrinsics1: Intrinsic matrix of the first camera.
    :param dist_coeffs1: Distortion coefficients of the first camera.
    :param intrinsics2: Intrinsic matrix of the second camera.
    :param dist_coeffs2: Distortion coefficients of the second camera.
    :param extrinsics1: 4x4 extrinsic matrix [R | t] of the first camera.
    :param extrinsics2: 4x4 extrinsic matrix [R | t] of the second camera.
    :return: Array of 3D points.
    """
    # Extract R and t from the extrinsic matrix
    R1, t1 = extrinsics1[:3, :3], extrinsics1[:3, 3:]
    R2, t2 = extrinsics2[:3, :3], extrinsics2[:3, 3:]

    # Undistort points
    points1_undistorted = cv2.undistortPoints(np.array(points1, dtype=np.float32), intrinsics1, dist_coeffs1)
    points2_undistorted = cv2.undistortPoints(np.array(points2, dtype=np.float32), intrinsics2, dist_coeffs2)

    # Triangulate points
    points_3d_hom = cv2.triangulatePoints(np.hstack((R1, t1)), np.hstack((R2, t2)), points1_undistorted, points2_undistorted)
    # Convert from homogeneous to 3D coordinates
    points_3d = cv2.convertPointsFromHomogeneous(points_3d_hom.T)

    return points_3d

def triangulate_points(pnts, extr, intr, output_dir = None):
    pnts1, pnts2 = pnts
    extrx1, extrx2 = extr
    intrinsics, dist = intr["intrinsics"], intr["dist"]
    pnts_3d = execute_triangulate_points(pnts1, pnts2, intrinsics, dist, intrinsics, dist, extrx1, extrx2)

    #### sanity check
    # err = calculate_reprojection_error(pnts_3d, pnts1, pnts2, intrinsics, dist, intrinsics, dist, extrx1, extrx2)
    # img1 = cv2.imread("/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/sofa_soccer_dragon_recon/images/01604.png")
    # img2 = cv2.imread("/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon/sofa_soccer_dragon_recon/images/01765.png")
    # prj_img1 = proj_3d_pnts(img1, intrinsics, extrx1, pnts_3d, dist_coeffs=dist)[-1]
    # prj_img2 = proj_3d_pnts(img2, intrinsics, extrx2, pnts_3d, dist_coeffs=dist)[-1]
    # assert 0
    if output_dir is not None:
        np.save(osp.join(output_dir, "triangulated.npy"), pnts_3d.squeeze())

    return pnts_3d


def calc_clearness_score(img_list, ignore_first = 0):
    """
    inp:
        img_list (list [str]): path to image files
    output:
        clear_img_fs (list [str]): sorted images files from clearest to blurriest
        best (list [int]): sorted idx of original img_list from clearest to blurriest
        blur_scores (list [float]): blur score of each images (higher is clearer)

    """
    # Get list of images in folder
    img_list = img_list[ignore_first:]

    # Load images
    images = parallel_map(imageio.imread, img_list, show_pbar=True, desc="loading imgs")

    blur_scores = []
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)
    blur_kernels = np.array([[
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ], [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], [
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ]], dtype=np.float32) / 5.0
    
    def calc_blur(image):
        gray_im = np.mean(image, axis=2)[::4, ::4]

        directional_blur_scores = []
        for i in range(4):
            blurred = ndimage.convolve(gray_im, blur_kernels[i])

            laplacian = signal.convolve2d(blurred, laplacian_kernel, mode="valid")
            var = laplacian**2
            var = np.clip(var, 0, 1000.0)

            directional_blur_scores.append(np.mean(var))

        antiblur_index = (np.argmax(directional_blur_scores) + 2) % 4

        return directional_blur_scores[antiblur_index]

    blur_scores = parallel_map(calc_blur, images, show_pbar=True, desc="calculating blur score")
    
    ids = np.argsort(blur_scores) + ignore_first
    best = ids[::-1]
 
    clear_image_fs = [img_list[e] for e in best]
    return clear_image_fs, best, np.array(blur_scores)


def project_points_radtan(points,
                          fx, fy, cx, cy,
                          k1, k2, p1, p2):
    
    """
    project point to plane
    """

    x_ = np.divide(points[:, :, 0], points[:, :, 2], out=np.zeros_like(points[:, :, 0]), where=points[:, :, 2]!=0)
    y_ = np.divide(points[:, :, 1], points[:, :, 2], out=np.zeros_like(points[:, :, 1]), where=points[:, :, 2]!=0)

    r2 = np.square(x_) + np.square(y_)
    r4 = np.square(r2)

    dist = (1.0 + k1 * r2 + k2 * r4)

    x__ = x_ * dist + 2.0 * p1 * x_ * y_ + p2 * (r2 + 2.0 * x_ * x_)
    y__ = y_ * dist + 2.0 * p2 * x_ * y_ + p1 * (r2 + 2.0 * y_ * y_)


    u = fx * x__ + cx
    v = fy * y__ + cy

    return u.astype(np.float32), v.astype(np.float32)