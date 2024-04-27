import numpy as np
import os.path as osp
import os
import glob
import cv2
import argparse
import glob
from tqdm import tqdm
import shutil

from nerfies.camera import Camera
import json

from sceneManager import DeblurRawSceneManager



def make_nerfies_camera(ext_mtx, intr_mtx, dist, img_size):
    """
    input:
        ext_mtx (np.array): World to cam matrix - shape = 4x4
        intr_mtx (np.array): intrinsic matrix of camera - shape = 3x3
        img_size [h, w] (list/tupple): size of image

    return:
        nerfies.camera.Camera of the given mtx
    """
    R = ext_mtx[:3,:3]
    t = ext_mtx[:3,3]
    k1, k2, p1, p2 = dist
    coord = -R.T@t  
    h, w = img_size

    cx, cy = intr_mtx[:2,2].astype(int)

    new_camera = Camera(
        orientation=R,
        position=coord,
        focal_length=intr_mtx[0,0],
        pixel_aspect_ratio=1,
        principal_point=np.array([cx, cy]),
        radial_distortion=(k1, k2, 0),
        tangential_distortion=(p1, p2),
        skew=0,
        image_size=np.array([w, h])  ## (width, height) of camera
    )

    return new_camera

def format_rgb_cameras(rgbScene:DeblurRawSceneManager, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(rgbScene)):
        M = rgbScene.get_extrnxs(i)
        K, dist = rgbScene.get_intrnxs()
        camera = make_nerfies_camera(M, K, dist, rgbScene.get_img_size())
        cam_json = camera.to_json()

        # if rgbScene.meta.get("mid_cam_ts") is not None:
        #     cam_json["t"] = rgbScene.get_camera_t(i)

        with open(osp.join(save_dir, f"{i:05d}.json"), "w") as f:
            json.dump(cam_json, f, indent=2)



def copy_imgs_to_dir(rgbScene:DeblurRawSceneManager, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    f_extension = osp.splitext(rgbScene.get_img_f(0))[-1]
    for i in tqdm(range(len(rgbScene)), desc="copying rgb images"):
        # img = rgbScene.get_img(i)
        # cv2.imwrite(osp.join(save_dir, f"{i:05d}.png"), img)
        shutil.copy(rgbScene.get_img_f(i), osp.join(save_dir, f"{i:05d}{f_extension}"))


def save_eimgs(evsScene, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    eimgs = evsScene.imgs.astype(np.int16)
    assert (np.abs(eimgs) < 127).all(), "can't format to int8!"
    eimgs = eimgs.astype(np.int8)

    save_f = osp.join(save_dir, "eimgs_1x.npy")
    np.save(save_f, eimgs)


def write_rgb_metadata(rgbScene:DeblurRawSceneManager, save_f):
    metadata = {"colmap_scale": rgbScene.get_colmap_scale()}

    for i in range(len(rgbScene)):
        metadata[str(i).zfill(5)] = {"warp_id": i,
                                     "appearance_id": i,
                                     "camera_id": 0}
    
    with open(save_f, "w") as f:
        json.dump(metadata, f, indent=2)


def write_dataset(rgbScene:DeblurRawSceneManager, save_f):

    ids = [str(i).zfill(5) for i in range(len(rgbScene))]
    test_ids = ids[::rgbScene.llffhold]
    train_ids = [i for i in ids if i not in test_ids]
    prep_ids, test_ids = test_ids[:1], test_ids[1:]

 
    dataset_json = {
        "count": len(ids),
        "num_exemplars": len(train_ids),
        "train_ids": train_ids,
        "val_ids": test_ids,
        "test_ids": test_ids,
        "prep_ids": prep_ids
    }

    with open(save_f, "w") as f:
        json.dump(dataset_json, f, indent=2)


def save_dummy_eimgs(save_dir, n_frames, h, w):
    os.makedirs(save_dir, exist_ok=True)
    eimgs = np.zeros((n_frames, h, w), dtype=np.int8)
    np.save(osp.join(save_dir, "eimgs_1x.npy"), eimgs)


def format_dummy_evs_cameras(save_dir, n_frames, h, w):
    os.makedirs(save_dir, exist_ok=True)
    M = np.eye(4)
    
    K = np.array([[100, 0, w//2],
                    [0, 100, h//2],
                    [0, 0, 1]])
    dist = np.zeros(4)

    for i in range(n_frames + 5):
        camera = make_nerfies_camera(M, K, dist, [h, w])
        cam_json = camera.to_json()

        with open(osp.join(save_dir, f"{i:05d}.json"), "w") as f:
            json.dump(cam_json, f, indent=2)


def write_dummy_metadata(save_f, n_frames):
    metadata = {}

    for i in range(n_frames):
        metadata[str(i).zfill(5)] = {"warp_id": i,
                                     "appearance_id": i,
                                     "camera_id": 0}
    
    with open(save_f, "w") as f:
        json.dump(metadata, f, indent=2)


def write_dummy_dataset(save_f, n_frames, n_digit = 5):
    ids = [str(i).zfill(n_digit) for i in range(n_frames)]
    test_ids = ids[::2]
    train_ids = [i for i in ids if i not in test_ids]
    prep_ids, test_ids = test_ids[:1], test_ids[1:]

    dataset_json = {
        "count": len(ids),
        "num_exemplars": len(train_ids),
        "train_ids": train_ids,
        "val_ids": test_ids,
        "test_ids": test_ids,
        "prep_ids": prep_ids
    }

    with open(save_f, "w") as f:
        json.dump(dataset_json, f, indent=2)

def main(scene_dir, targ_dir=None):
    if targ_dir is None:
        targ_dir = osp.join("/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data", osp.basename(scene_dir))
    colcam_dir = osp.join(targ_dir, "colcam_set")
    ecam_dir = osp.join(targ_dir, "ecam_set")
    rgbScene = DeblurRawSceneManager(scene_dir)

    
    save_img_dir = osp.join(colcam_dir, "rgb", "1x")
    copy_imgs_to_dir(rgbScene, save_img_dir)

    save_rgb_cam_dir = osp.join(colcam_dir, "camera")
    format_rgb_cameras(rgbScene, save_rgb_cam_dir)

    rgb_metadata_f = osp.join(colcam_dir, "metadata.json")
    write_rgb_metadata(rgbScene, rgb_metadata_f)

    rgb_dataset_f = osp.join(colcam_dir, "dataset.json")
    write_dataset(rgbScene, rgb_dataset_f)



    n_frames, h, w = 10, 128, 128  # dummy params

    save_eimgs_dir = osp.join(ecam_dir, "eimgs")
    save_dummy_eimgs(save_eimgs_dir, n_frames, h, w)

    save_evs_cam_dir = osp.join(ecam_dir, "camera")
    format_dummy_evs_cameras(save_evs_cam_dir, n_frames, h, w)

    write_dummy_metadata(osp.join(ecam_dir, "metadata.json"), n_frames)

    write_dummy_dataset(osp.join(ecam_dir, "dataset.json"), n_frames)


if __name__ == "__main__":
    scene_dir = "/ubc/cs/research/kmyi/matthew/projects/Deblur-NeRF/data/blurparterre"
    targ_dir = osp.join("/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data", osp.basename(scene_dir))
    main(scene_dir, targ_dir)