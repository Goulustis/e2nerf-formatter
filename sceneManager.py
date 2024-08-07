import os.path as osp
import glob
import numpy as np
import json
import torch
import cv2

from camera_utils import load_poses_bounds, poses_to_w2cs_hwf, load_json_cam, load_json_intr
from misc_utils import get_llffhold




class E2NerfRGBManager:

    def __init__(self, data_dir, single_cam=True):
        self.data_dir = data_dir
        self.single_cam = single_cam
        self.img_fs = sorted(glob.glob(osp.join(data_dir, "images", "*")))
        
        self.mid_rgb_poses_bounds_f = osp.join(data_dir, "mid_rgb_poses_bounds.npy")
        rgb_poses_bounds_f = osp.join(data_dir, "rgb_poses_bounds.npy") if not osp.exists(self.mid_rgb_poses_bounds_f) else self.mid_rgb_poses_bounds_f
        if osp.exists(rgb_poses_bounds_f):
            self.poses, self.bds, self.hwf = load_poses_bounds(rgb_poses_bounds_f)
        else:
            self.poses, self.bds, self.hwf = load_poses_bounds(osp.join(data_dir, "poses_bounds.npy"))
        self.w2cs, _ = poses_to_w2cs_hwf(self.poses)
        self.w2cs = self.w2cs[:, :3, :4]
        self.hwf = self.hwf[..., 0]

        self.n_bins = 5
        meta_f = osp.join(data_dir, "metadata.json")
        self.meta = None
        if osp.exists(meta_f):
            with open(meta_f, "r") as f:
                self.meta = json.load(f)
            
            self.n_bins = self.meta["n_bins"]
        
        self.ori_w2cs = self.w2cs
        self.w2cs = self.w2cs.reshape(-1, self.n_bins, 3, 4) if not osp.exists(self.mid_rgb_poses_bounds_f) else self.w2cs
        if single_cam and (not osp.exists(self.mid_rgb_poses_bounds_f)):
            self.w2cs = self.w2cs[:, self.n_bins//2, :, :]
        
        self.img_size = self.get_img(0).shape[:2]

    def __len__(self):
        if hasattr(self, "meta") and self.meta.get("mid_cam_ts") is not None:
            return len(self.meta.get("mid_cam_ts"))

        return len(self.img_fs)
        # return min(len(self.imgs), len(self.w2cs))
    
    # REQUIRED
    def get_img_size(self):
        return self.img_size # (h, w)
    
    # REQUIRED
    def get_img_f(self, idx):
        return self.img_fs[idx]

    # REQUIRED
    def get_img(self, idx):
        return cv2.imread(self.img_fs[idx])
    
    # REQUIRED
    def get_extrnxs(self, idx):
        return self.w2cs[idx]

    def get_camera_t(self, idx):
        return self.meta["mid_cam_ts"][idx]
    
    def get_colmap_scale(self):
        return self.meta.get("colmap_scale")

    # REQUIRED
    def get_intrnxs(self):
        if self.meta is None or self.meta.get("rgb_K") is None:
            return np.array([[self.hwf[2], 0, self.hwf[1]/2],
                            [0, self.hwf[2], self.hwf[0]/2],
                            [0,           0,           1]]), np.zeros(4)
        else:
            return np.array([[self.hwf[2], 0, self.meta["rgb_K"][2]],
                             [0, self.hwf[2], self.meta["rgb_K"][3]],
                             [0,           0,           1          ]]), np.zeros(4)


class E2NeRFEVSManager(E2NerfRGBManager):
    def __init__(self, data_dir, single_cam=True):
        """
        data_dir (str): path to scene
        single_cam (bool): cams are saved as (n, bin_size, *), if true, will select a single eimg and camera with the shape (n, *)
        """
        self.data_dir = data_dir
        self.single_cam = single_cam
        self.imgs = torch.load(osp.join(self.data_dir, "events.pt")).numpy()

        evs_poses_bounds_f = osp.join(data_dir, "evs_poses_bounds.npy")
        if osp.exists(evs_poses_bounds_f):
            self.poses, self.bds, self.hwf = load_poses_bounds(evs_poses_bounds_f)
        else:
            self.poses, self.bds, self.hwf = load_poses_bounds(osp.join(data_dir, "poses_bounds.npy"))
        self.w2cs, _ = poses_to_w2cs_hwf(self.poses)
        self.w2cs = self.w2cs[:, :3, :4]
        self.hwf = self.hwf[..., 0]


        self.n_bins = 5
        meta_f = osp.join(data_dir, "metadata.json")
        self.meta = None
        if osp.exists(meta_f):
            with open(meta_f, "r") as f:
                self.meta = json.load(f)
            self.n_bins = self.meta["n_bins"]
        
        h, w = self.hwf[:2].astype(int)
        self.img_size = (h, w)

        self.w2cs = self.w2cs.reshape(-1, self.n_bins, 3, 4)
        self.imgs = self.imgs.reshape(-1, self.n_bins - 1, h, w)

        if single_cam:
            self.w2cs = self.w2cs[:, self.n_bins//2, :, :]
            self.imgs = self.imgs[:, self.n_bins//2, :, :]
        else:
            self.imgs = self.imgs.reshape(-1, h, w)
        
        ev_t = self.meta.get("ev_cam_ts")
        self.cam_t = np.array(self.meta["ev_cam_ts"]).reshape(-1, self.n_bins) if not (ev_t is None) else None

    def __len__(self):
        return min(len(self.imgs), len(self.w2cs))

    def get_img_f(self, idx):
        assert 0, "Not implemented"
    
    def get_img(self, idx):
        return np.stack([(self.imgs[idx] != 0).astype(np.uint8) * 255]*3, axis=-1)

    def get_intrnxs(self):
        if self.meta is None or self.meta.get("evs_K") is None:
            return super().get_intrnxs()
        
        return np.array([[self.hwf[2], 0, self.meta["evs_K"][2]],
                         [0, self.hwf[2], self.meta["evs_K"][3]],
                         [0,           0,           1]]), np.zeros(4)


class ColcamManager(E2NerfRGBManager):
    def __init__(self, data_dir) -> None:
        data_dir = osp.join(data_dir, "colcam_set")
        self.data_dir = data_dir
        self.img_fs = sorted(glob.glob(osp.join(self.data_dir, "rgb", "1x", "*.png")))
        self.imgs = np.stack([cv2.imread(img_f) for img_f in self.img_fs])
        
        self.cam_fs = sorted(glob.glob(osp.join(self.data_dir, "camera", "*.json")))
        self.K, self.D = load_json_intr(self.cam_fs[0])

        self.w2cs = [load_json_cam(cam_f) for cam_f in self.cam_fs]
    
    def get_intrnxs(self):
        return self.K, self.D


class EcamManager(E2NeRFEVSManager):

    def __init__(self, data_dir) -> None:
        data_dir = osp.join(data_dir, "ecam_set")
        self.data_dir = data_dir

        self.imgs = np.load(osp.join(data_dir, "eimgs", "eimgs_1x.npy"))
        self.cam_fs = sorted(glob.glob(osp.join(data_dir, "camera", "*.json")))
        # self.cam_fs = sorted(glob.glob(osp.join(data_dir, "prev_camera", "*.json")))

        self.K, self.D = load_json_intr(self.cam_fs[0])
        self.w2cs = [load_json_cam(cam_f) for cam_f in self.cam_fs]

    def get_img_f(self, idx):
        assert 0, "Not implemented"
    

    def get_intrnxs(self):
        return self.K, self.D
    

class DeblurRawSceneManager(E2NerfRGBManager):
    def __init__(self, data_dir):
        DEBLUR_NERF_CONFIG_DIR = "/ubc/cs/research/kmyi/matthew/projects/Deblur-NeRF/configs"
        config_f = glob.glob(osp.join(DEBLUR_NERF_CONFIG_DIR, osp.basename(data_dir), "*.txt"))[0]

        self.data_dir = data_dir
        self.llffhold = get_llffhold(config_f)
        self.img_fs = sorted(glob.glob(osp.join(data_dir, "images", "*")))

        poses_bounds_f = osp.join(data_dir, "poses_bounds.npy")
        self.poses, self.bds, self.hwf = load_poses_bounds(poses_bounds_f)
        self.w2cs, _ = poses_to_w2cs_hwf(self.poses)
        self.w2cs = self.w2cs[:, :3, :4]
        self.hwf = self.hwf[..., 0]

        self.meta = {}
        self.img_size = self.get_img(0).shape[:2]
    
    def get_intrnxs(self):
        return np.array([[self.hwf[2], 0, self.hwf[1]/2],
                        [0, self.hwf[2], self.hwf[0]/2],
                        [0,           0,           1]]), np.zeros(4)