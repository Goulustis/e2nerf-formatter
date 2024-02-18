import os.path as osp
import glob
import numpy as np
import json
import torch
import cv2

from camera_utils import load_poses_bounds, poses_to_w2cs_hwf




class E2NerfRGBManager:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_fs = sorted(glob.glob(osp.join(data_dir, "images", "*")))
        
        rgb_poses_bounds_f = osp.join(data_dir, "rgb_poses_bounds.npy")
        if osp.exists(rgb_poses_bounds_f):
            self.poses, self.bds, self.hwf = load_poses_bounds(rgb_poses_bounds_f)
        else:
            self.poses, self.bds, self.hwf = load_poses_bounds(osp.join(data_dir, "poses_bounds.npy"))
        self.w2cs, _ = poses_to_w2cs_hwf(self.poses)
        self.w2cs = self.w2cs[:, :3, :4]
        self.hwf = self.hwf[..., 0]

        n_bins = 5
        meta_f = osp.join(data_dir, "metadata.json")
        self.meta = None
        if osp.exists(meta_f):
            with open(meta_f, "r") as f:
                self.meta = json.load(f)
            
            n_bins = self.meta["n_bins"]
        self.w2cs = self.w2cs.reshape(-1, n_bins, 3, 4)[:, n_bins//2, :, :]

    def __len__(self):
        return len(self.img_fs)

    def get_img_f(self, idx):
        return self.img_fs[idx]

    def get_img(self, idx):
        return cv2.imread(self.img_fs[idx])
    
    def get_extrnxs(self, idx):
        return self.w2cs[idx]

    def get_intrnxs(self):
        return np.array([[self.hwf[2], 0, self.hwf[1]/2],
                         [0, self.hwf[2], self.hwf[0]/2],
                         [0,           0,           1]]), np.zeros(4)


class E2NeRFEVSManager(E2NerfRGBManager):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = torch.load(osp.join(self.data_dir, "events.pt")).numpy()
        # self.imgs = np.load(osp.join(self.data_dir, "events.npy"))

        evs_poses_bounds_f = osp.join(data_dir, "evs_poses_bounds.npy")
        if osp.exists(evs_poses_bounds_f):
            self.poses, self.bds, self.hwf = load_poses_bounds(evs_poses_bounds_f)
        else:
            self.poses, self.bds, self.hwf = load_poses_bounds(osp.join(data_dir, "poses_bounds.npy"))
        self.w2cs, _ = poses_to_w2cs_hwf(self.poses)
        self.w2cs = self.w2cs[:, :3, :4]
        self.hwf = self.hwf[..., 0]


        n_bins = 5
        meta_f = osp.join(data_dir, "metadata.json")
        self.meta = None
        if osp.exists(meta_f):
            with open(meta_f, "r") as f:
                self.meta = json.load(f)
            n_bins = self.meta["n_bins"]
        
        h, w = self.hwf[:2].astype(int)
        self.w2cs = self.w2cs.reshape(-1, n_bins, 3, 4)[:, n_bins//2, :, :]
        self.imgs = self.imgs.reshape(-1, n_bins - 1, h, w)[:, n_bins//2, :, :]

    def __len__(self):
        return len(self.imgs)

    def get_img_f(self, idx):
        assert 0, "Not implemented"
    
    def get_img(self, idx):
        return np.stack([(self.imgs[idx] != 0).astype(np.uint8) * 255]*3, axis=-1)

    def get_intrnxs(self):
        if self.meta is None:
            return super().get_intrnxs()
        
        return np.array([[self.hwf[2], 0, self.meta["evs_K"][2]],
                         [0, self.hwf[2], self.meta["evs_K"][3]],
                         [0,           0,           1]]), np.zeros(4)