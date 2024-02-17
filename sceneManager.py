import os.path as osp
import glob
import numpy as np
import json
import torch
import cv2

from nerfies.camera import Camera
from camera_utils import load_poses_bounds, poses_to_w2cs_hwf


class ColcamSeneManager:

    def __init__(self, colcam_set_dir):
        self.colcam_set_dir = colcam_set_dir
        self.img_fs = sorted(glob.glob(osp.join(colcam_set_dir, "rgb", "1x", "*.png")))
        self.cam_fs = sorted(glob.glob(osp.join(colcam_set_dir, "camera", "*.json")))

        self.ref_cam = Camera.from_json(self.cam_fs[0])


    def get_img_f(self,idx):
        return self.img_fs[idx]

    def get_extrnxs(self, idx):
        """
        returns:
            world-to-camera transformation matrix
        """
        extrxs_f = self.cam_fs[idx]
        cam = Camera.from_json(extrxs_f)
        R = cam.orientation
        T = -R@cam.position
        T = T.reshape(3,1)

        return np.concatenate([R, T], axis=1)

    def get_intrnxs(self):
        """
        return K, distortions
        """
        fx = fy = self.ref_cam.focal_length
        cx, cy = self.ref_cam.principal_point_x, self.ref_cam.principal_point_y
        k1, k2, k3 = self.ref_cam.radial_distortion
        p1, p2 = self.ref_cam.tangential_distortion

        intrx_mtx = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
        dist = np.array((k1, k2, p1, p2))

        return intrx_mtx, dist


class E2NerfRGBManager:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_fs = sorted(glob.glob(osp.join(data_dir, "images", "*.png")))
        
        self.poses, self.bds, self.hwf = load_poses_bounds(osp.join(data_dir, "rgb_poses_bounds.npy"))
        self.w2cs, _ = poses_to_w2cs_hwf(self.poses)

        meta_f = osp.join(data_dir, "meta.json")
        with open(meta_f, "r") as f:
            self.meta = json.load(f)
        
        n_bins = self.meta["n_bins"]
        self.w2cs = self.w2cs.reshape(-1, n_bins, 3, 4)[:, n_bins//2, :, :]


    def get_img_f(self, idx):
        return self.img_fs[idx]

    def get_img(self, idx):
        return cv2.imread(self.img_fs[idx])
    
    def get_extrnxs(self, idx):
        return self.w2cs[idx]

    def get_intrnxs(self):
        return np.array([[self.hwf[2], 0, self.hwf[0]],
                         [0, self.hwf[2], self.hwf[1]],
                         [0,           0,           1]]), np.array([0,0,0,0])


class E2NeRFEVSManager(E2NerfRGBManager):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = torch.load(osp.join(self.data_dir, "events.pt"))
        self.poses, self.bds, self.hwf = load_poses_bounds(osp.join(data_dir, "evs_poses_bounds.npy"))
        self.w2cs, _ = poses_to_w2cs_hwf(self.poses)

        meta_f = osp.join(data_dir, "meta.json")
        with open(meta_f, "r") as f:
            self.meta = json.load(f)
        
        h, w = self.hwf[...,0][:2]
        n_bins = self.meta["n_bins"]
        self.w2cs = self.w2cs.reshape(-1, n_bins, 3, 4)[:, n_bins//2, h, w]


    def get_img_f(self, idx):
        assert 0, "Not implemented"
    
    def get_img(self, idx):
        return (self.imgs[idx] != 0).astype(np.uint8)