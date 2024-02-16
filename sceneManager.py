import os.path as osp
import glob
import numpy as np

from nerfies.camera import Camera



class ColcamSeneManager:

    def __init__(self, colcam_set_dir):
        self.colcam_set_dir = colcam_set_dir
        self.img_fs = sorted(glob.glob(osp.join(colcam_set_dir, "rgb", "1x", "*.png")))
        self.cam_fs = sorted(glob.glob(osp.join(colcam_set_dir, "camera", "*.json")))

        self.ref_cam = Camera.from_json(self.cam_fs[0])


    def get_img_f(self,idx):
        return self.img_fs[idx]

    def get_extrnxs(self, idx):
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