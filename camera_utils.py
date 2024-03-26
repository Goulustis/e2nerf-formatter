import cv2
import json
import numpy as np
from llff.poses.colmap_read_model import read_cameras_binary, read_images_binary

# reference:
# https://github.com/colmap/colmap/blob/8e7093d22b324ce71c70d368d852f0ad91743808/src/colmap/sensor/models.h#L268C34-L268C34
def read_colmap_cam(cam_path, scale=1):
    # fx, fy, cx, cy, k1, k2, k3, k4
    cam = read_cameras_binary(cam_path)
    cam = cam[1]

    fx, fy, cx, cy = cam.params[:4]*scale
    int_mtx = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]]) 
    
    # cv2 model fx, fy, cx, cy, k1, k2, p1, p2
    return {"M1": int_mtx, "d1":cam.params[4:]}

def read_colmap_cam_param(cam_path):
    out = read_colmap_cam(cam_path, scale=1)

    return out["M1"], out["d1"]

def read_prosphesee_ecam_param(cam_path):
    
    with open(cam_path,"r") as f:
        data = json.load(f)
    
    return np.array(data["camera_matrix"]['data']).reshape(3,3), \
           np.array(data['distortion_coefficients']['data'])


def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Undistorts an image using the given camera matrix and distortion coefficients.

    Parameters:
    - image: The distorted input image.
    - camera_matrix: The camera intrinsic matrix.
    - dist_coeffs: The distortion coefficients (k1, k2, p1, p2).

    Returns:
    - undistorted_image: The undistorted output image.
    """

    # Get the image size
    h, w = image.shape[:2]

    # Calculate the undistortion and rectification transformation map
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5)

    # Remap the original image to the new undistorted image
    undist_img = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # Crop the image to the ROI
    x, y, w, h = roi
    undist_img = undist_img[y:y+h, x:x+w]

    return undist_img


def poses_to_w2cs_hwf(poses):
    """
    takes LLFF poses and return to colmap c2w
    poses (3x5xn)
    """

    inv = np.concatenate([poses[:,1:2,:], poses[:,0:1,:], -poses[:,2:3,:], poses[:, 3:]],1)
    c2ws = inv[:3,:4,:]
    dummy = np.zeros((1,4,1))
    dummy[0,-1,0] = 1
    c2ws = np.concatenate([c2ws, np.tile(dummy, (1, 1, c2ws.shape[-1]))]).transpose(2,0,1)
    w2cs = np.linalg.inv(c2ws)

    ## w2cs, hwf
    return w2cs, poses[:,4:,:]



def w2cs_hwf_to_poses(w2c_mats, hwf):
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, hwf], 1)

    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    return poses


def load_poses_bounds(path):
    poses_arr = np.load(path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    hwf = poses[:, 4:, :].squeeze()

    # shapes = (3,5,n), (2,n), (3,n)
    return poses, bds, hwf


def load_json_intr(cam_f):
    with open(cam_f, "r") as f:
        data = json.load(f)
        fx = fy = data["focal_length"]
        cx, cy = data["principal_point"]
        k1, k2, k3 = data["radial_distortion"]
        p1, p2 = data["tangential_distortion"]
    
    return np.array([[fx, 0, cx],
                    [0,   fy, cy],
                    [0, 0, 1]]), np.array((k1,k2,p1,p2))



def load_json_cam(cam_f):
    with open(cam_f, "r") as f:
        data = json.load(f)
        R, pos = np.array(data["orientation"]), np.array(data["position"])
        t = -(pos@R.T).T
        t = t.reshape(-1,1)
    
    return np.concatenate([R, t], axis=-1)

if __name__ == "__main__":
    colcam_path = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/calib_checker_recons/sparse/0/cameras.bin"
    img_f = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/black_seoul_b3_v3/colcam_set/rgb/1x/00000.png"
    K, D = read_colmap_cam_param(colcam_path)
    img = cv2.imread(img_f)
    undist = undistort_image(img, K, D)
    
