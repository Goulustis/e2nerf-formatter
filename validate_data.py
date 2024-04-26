from tqdm import tqdm
import json
import cv2
import numpy as np
import glob
import os.path as osp
import os
import matplotlib.pyplot as plt
import contextlib


from pointselector import ImagePointSelector
from misc_utils import parallel_map
from proj_utils import pnp_extrns, triangulate_points, proj_3d_pnts, calc_clearness_score
from sceneManager import E2NerfRGBManager, E2NeRFEVSManager, ColcamManager, EcamManager
from edllff_loader import EdllffEVSWrapper, EdllffRGBWrapper

TMP_DIR = "./tmp"




def get_img_fs(rgb_dir, eimg_dir, n_use = 300, st_n=150):
    sub_sample = lambda x : x[::len(x)//n_use]

    images_to = sorted(glob.glob(osp.join(rgb_dir, "*")))[st_n:]
    images_from = sorted(glob.glob(osp.join(eimg_dir, "*")))[st_n:]

    n_frames = min(len(images_from), len(images_to))
    images_from = images_from[:n_frames]
    images_to = images_to[:n_frames]

    images_to, images_from = sub_sample(images_to), sub_sample(images_from)

    return images_to, images_from

def get_calib_objpnts(grid_size=3):
    row, col = 5,8   # inner cornors of the checker, see https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
    objp = np.zeros((row*col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2)
    objp = objp*grid_size

    return objp

def find_all_extrnsics(imgs, K, D, pnts_2d=None):
    objpnts = get_calib_objpnts(3)

    pnts_2d = parallel_map(lambda x: cv2.findChessboardCorners(x, (5, 8), None), imgs, show_pbar=True, desc="finding corners") #(r, c)
    cond = np.stack(e[0] for e in pnts_2d)
    pnts_2d = np.stack([e[1] for e in pnts_2d])[cond]


    extrs = []
    for pnt in pnts_2d:
        R, t = pnp_extrns(objpnts, pnt, K, D)
        extrs.append(np.concatenate([R, t], axis=1))
    
    return extrs, cond


def load_relcam(relcam_f):
    with open(relcam_f, "r") as f:
        data = json.load(f)
    
    return np.array(data["M1"]), np.array(data["dist1"]), np.array(data["M2"]), np.array(data["dist2"]), np.array(data["R"]), np.array(data["T"])




def find_chessboard_corners(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, pnts = cv2.findChessboardCorners(img, (5,8), None)
    assert ret

    criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(img, pnts, (11,11),(-1,-1), criteria)
    return pnts

def load_objpnts(colmap_pnts_f, colmap_dir=None, calc_clear=False, use_checker=False):
    if osp.exists(colmap_pnts_f):
        colmap_pnts = np.load(colmap_pnts_f)
    else:
        assert colmap_dir is not None, "need all other params to create 3d points"
        # manager = E2NerfRGBManager(colmap_dir)
        manager = ColcamManager(colmap_dir)
        rgb_K, rgb_D = manager.get_intrnxs()

        # takes a formated dataset; so the index starts at 0 now
        if calc_clear:
            clear_idxs = calc_clearness_score([manager.get_img_f(i) for i in range(len(manager))])[1]
            idx1, idx2 = clear_idxs[0], clear_idxs[0 + 3]            
        else:
            idx1, idx2 = 0, 35
            # idx1, idx2 = 105, 116

        selector = ImagePointSelector([manager.get_img_f(idx1), manager.get_img_f(idx2)], save_dir=TMP_DIR)
        # selector = ImagePointSelector([manager.get_img(idx1), manager.get_img(idx2)], save_dir=TMP_DIR)        
        if not use_checker:
            pnts = selector.select_points()
        else:
            pnts = selector.select_checker()
            
        colmap_pnts = triangulate_points(pnts, [manager.get_extrnxs(idx1), manager.get_extrnxs(idx2)], 
                                        {"intrinsics": rgb_K, "dist":rgb_D}, TMP_DIR)
        np.save(colmap_pnts_f, colmap_pnts)
    
    return colmap_pnts


def load_json_cam(cam_f):
    with open(cam_f, "r") as f:
        data = json.load(f)
        R, pos = np.array(data["orientation"]), np.array(data["position"])
        t = -(pos@R.T).T
        t = t.reshape(-1,1)
    
    return np.concatenate([R, t], axis=-1)


def load_json_intr(cam_f):
    with open(cam_f, "r") as f:
        data = json.load(f)
        fx = fy = data["focal_length"]
        cx, cy = data["principal_point"]
        k1, k2, k3 = data["radial_distortion"]
        p1, p2 = data["tangential_distortion"]
    
    return np.array([[fx, 0, cx],
                    [0,   fy, cy],
                    [0, 0, 1]]), (k1,k2,p1,p2)


### IMPLEMENTATION IS CORRECT FOR VALIDATION
def validate_ecamset():

    # ecamset = "/ubc/cs/research/kmyi/matthew/projects/E2NeRF/data/real-world/cs_building_v6"
    ecamset = "/ubc/cs/research/kmyi/matthew/projects/evimo-formatter/debug"
    # ecamset = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/depth_var_1_lr_000000"
    # ecamset = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/racoon_v6_rect"
    colmap_dir = ecamset

    scene = osp.basename(ecamset)
    objpnts_f = f"tmp/{scene}_triangulated.npy"

    save_dir = osp.join(TMP_DIR, f"{osp.basename(ecamset)}_proj")

    os.makedirs(save_dir, exist_ok=True)

    # evsManager = ColcamManager(ecamset)
    evsManager = EcamManager(ecamset)
    # evsManager = E2NeRFEVSManager(ecamset)
    # evsManager = E2NerfRGBManager(ecamset)
    # evsManager = EdllffRGBWrapper(ecamset)
    # evsManager = EdllffEVSWrapper(ecamset)

    ecam_K, ecam_D = evsManager.get_intrnxs()
    ecams = parallel_map(evsManager.get_extrnxs, list(range(len(evsManager))), show_pbar=True, desc="loading evs extrinsics")


    objpnts = load_objpnts(objpnts_f, colmap_dir, calc_clear=False, use_checker=False)

    def proj_fn(inp):
        img, extr = inp
        return proj_3d_pnts(img, ecam_K, extr, objpnts, dist_coeffs=ecam_D)[1]


    eimgs = parallel_map(evsManager.get_img, list(range(len(evsManager))), show_pbar=True, desc="loading evs imgs")
    proj_eimgs = parallel_map(proj_fn, list(zip(eimgs, ecams)), show_pbar=True, desc="projecting points")

    
    def save_fn(inp):
        img, idx = inp
        flag = cv2.imwrite(osp.join(save_dir, f"{str(idx).zfill(6)}.png"), img)
        assert flag, "save img failed"
    
    parallel_map(save_fn, list(zip(proj_eimgs, list(range(len(eimgs))))), 
                 show_pbar=True, desc="saving projected")

    save_f = f"{TMP_DIR}/{scene}_{osp.basename(ecamset)}.mp4"

    with contextlib.suppress(FileNotFoundError):
        os.remove(save_f)

    # os.system(f"ffmpeg -framerate 16 -i {save_dir}/%06d.png -c:v libx264 -pix_fmt yuv420p -frames:v 7500 {save_f}")
    os.system(f"ffmpeg -framerate 16 -i {save_dir}/%06d.png -c:v h264_nvenc -preset fast -pix_fmt yuv420p -frames:v 6050 {save_f}")
    # os.system(f"ffmpeg -framerate 16 -i {save_dir}/%06d.png -vf \"drawtext=fontfile=/path/to/font.ttf: text='%{{frame_num}}': start_number=0: x=10: y=10: fontcolor=white: fontsize=24: box=1: boxcolor=black@0.5: boxborderw=5\" -c:v libx264 -pix_fmt yuv420p vid.mp4")
    print("saved to", save_f)


if __name__ == "__main__":
    validate_ecamset()