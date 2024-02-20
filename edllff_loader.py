import torch
import numpy as np
import imageio
import os.path as osp
import glob
import json

from camera_utils import load_poses_bounds
from misc_utils import parallel_map

# def e2nerf_poses_prep(poses, bds):
#     poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
#     poses = np.moveaxis(poses, -1, 0).astype(np.float32)
#     bds = np.moveaxis(bds, -1, 0).astype(np.float32)

#     return poses[..., :4], bds


def e2nerf_poses_prep(poses, bds):
    inv = np.concatenate([poses[:,1:2,:], poses[:,0:1,:], -poses[:,2:3,:], poses[:, 3:]],1)
    # inv = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    c2ws = inv[:3,:4,:]
    dummy = np.zeros((1,4,1))
    dummy[0,-1,0] = 1
    c2ws = np.concatenate([c2ws, np.tile(dummy, (1, 1, c2ws.shape[-1]))]).transpose(2,0,1)
    w2cs = np.linalg.inv(c2ws)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    return w2cs, bds

def load_nerf_poses_bounds(path):
    """
    load poses_bounds such that it is compatible with e2nerf, it does somthing fancy
    returns:
        poses (n, 3, 5)
        bds (n, 2)
    
    """

    poses, bds, hwf = load_poses_bounds(path)
    poses, bds = e2nerf_poses_prep(poses, bds)
    return poses, bds, hwf.T


def load_json(path):    
    with open(path, "r") as f:
        return json.load(f)


class EdllffLoader:

    def __init__(self, datadir) -> None:
        self.datadir = datadir

        self.imgs = self._load_all_rgbs()
        self.meta = load_json(osp.join(self.datadir, "metadata.json"))
        self.dataset = load_json(osp.join(self.datadir, "dataset.json"))
        self.n_bins = self.meta["n_bins"]


    def _load_all_rgbs(self):
       img_fs = sorted(glob.glob(osp.join(self.datadir, "images", "*.png")))
       imgs = np.stack(parallel_map(imageio.imread, img_fs, show_pbar=True, desc="Loading images"))
       imgs = imgs.astype(np.float32) / 255
       return imgs


    def load_rgb_train(self):
        """
        returns:
            train_imgs (n, h, w, 3)
            train_poses (n, n_bins, 3, 4), 
            bds (n*n_bins, 2)
        """
        train_ids = np.array([int(i) for i in self.dataset["train_ids"]])
        train_imgs = np.stack([self.imgs[i] for i in train_ids])

        poses, bds, hwf = load_nerf_poses_bounds(osp.join(self.datadir, "rgb_poses_bounds.npy"))
        poses = poses.reshape(-1, self.meta["n_bins"], *poses.shape[-2:])

        self.rgb_hwf = hwf[0]

        return self.imgs, poses, bds, hwf
        # return train_imgs, poses, bds, hwf
        # poses = poses[train_ids]


    def get_n_bins(self):
        return self.n_bins
    
    def get_rgb_shape(self):
        return self.imgs[0].shape[:2]

    def load_rgb_test(self):
        """
        returns:
            test_imgs (n, h, w, 3)
            test_poses (n, 3, 4)
        """
        test_ids = np.array([int(i) for i in self.dataset["test_ids"]])
        test_imgs = np.stack([self.imgs[i] for i in test_ids])

        poses, bds, hwf = load_nerf_poses_bounds(osp.join(self.datadir, "mid_rgb_poses_bounds.npy"))
        poses = poses[test_ids]
        poses = poses[:, None]  # n_bins = 1

        self.rgb_hwf = hwf[0]

        return test_imgs, poses, bds, hwf
        
    
    def load_event_train(self):
        """
        returns:
            eimgs (n, n_bins-1, h*w)
            evs_poses (n, n_bins, 3, 4)
        """

        evs_f = osp.join(self.datadir, "events.pt")
        self.eimgs = torch.load(evs_f)
        evs_poses, bds, hwf = load_nerf_poses_bounds(osp.join(self.datadir, "evs_poses_bounds.npy"))
        evs_poses = evs_poses.reshape(-1, self.meta["n_bins"], *evs_poses.shape[-2:])
        self.evs_hwf = hwf[0]

        return self.eimgs, evs_poses

    def gen_evs_combination(self):
        combination = []
        for m in range(self.meta["n_bins"] - 1):
            for n in range(m + 1, self.meta["n_bins"]):
                combination.append([m, n])

        return combination
    


    
    def get_rgb_intrinsics(self):
        """
        returns:
            intrinsics (3, 3)
        """
        return np.array([[self.rgb_hwf[2], 0, self.meta["rgb_K"][2]],
                         [0, self.rgb_hwf[2], self.meta["rgb_K"][3]],
                         [0,           0,           1          ]])

    
    def get_evs_intrinsics(self):
        """
        returns:
            intrinsics (3, 3)
        """

        return np.array([[self.evs_hwf[2], 0, self.meta["evs_K"][2]],
                         [0, self.evs_hwf[2], self.meta["evs_K"][3]],
                         [0,           0,           1]])

    def get_evs_shape(self):
        return self.meta["evs_hw"]


class EdllffRGBWrapper:
    def __init__(self, data_dir) -> None:
        self.loader = EdllffLoader(data_dir)
        self.images, self.poses, _, _ = self.loader.load_rgb_train()
        
        self.n_bins = self.poses.shape[1]
        self.poses = self.poses[:, self.n_bins//2]

        self.img_size = self.images[0].shape[:2]
    
    def __len__(self):
        return len(self.images)

    def get_img_size(self):
        return self.img_size

    def get_img_f(self, idx):
        assert 0, "not supported"
    
    def get_img(self, idx):
        return (self.images[idx] * 255).astype(np.uint8)

    def get_extrnxs(self, idx):
        return self.poses[idx]

    def get_intrnxs(self):
        return self.loader.get_rgb_intrinsics().astype(np.float32), np.zeros(4).astype(np.float32)


class EdllffEVSWrapper:
    def __init__(self, data_dir) -> None:
        self.loader = EdllffLoader(data_dir)
        self.images, self.evs_poses = self.loader.load_event_train()

        h, w = self.loader.evs_hwf[:2].astype(int)
        self.images = self.images.reshape(*self.images.shape[:2], h, w)
        
        self.n_bins = self.evs_poses.shape[1]
        self.poses = self.evs_poses[:, self.n_bins//2]
        self.images = self.images[:, self.n_bins//2]
        self.img_size = self.images[0].shape[:2]
    
    def __len__(self):
        return len(self.images)

    def get_img_size(self):
        return self.img_size

    def get_img_f(self, idx):
        assert 0, "not supported"
    
    def get_img(self, idx):
        return (np.stack([self.images[idx] != 0] * 3, axis = -1)*255).astype(np.uint8)

    def get_extrnxs(self, idx):
        return self.poses[idx]

    def get_intrnxs(self):
        return self.loader.get_evs_intrinsics().astype(np.float32), np.zeros(4).astype(np.float32)



class EllffLoader:

    def __init__(self, datadir) -> None:
        self.datadir = datadir
    
    def load_data(self):
        """
        returns:
            imgs (n, h, w, 3)
            poses (n, 3, 5)
            bds (n, 2)
            hwf (3,)
        """
        imgs = np.stack(parallel_map(imageio.imread, sorted(glob.glob(osp.join(self.datadir, "images", "*"))), show_pbar=True, desc="Loading images"))/255
        poses, bds, _ = load_nerf_poses_bounds(osp.join(self.datadir, "poses_bounds.npy"))
        poses = poses.reshape(30, 5, 3, 4)
        return imgs, poses, bds
