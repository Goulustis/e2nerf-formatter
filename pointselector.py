import cv2
import numpy as np
import os
import os.path as osp

def detect_chessboard(img):
    ret, pnts = cv2.findChessboardCorners(img, (5, 8), None)
    assert ret
    return pnts

class ImagePointSelector:
    def __init__(self, image_paths, show_point_indices=True, save=True, save_dir = None, save_fs = None, end_fix="pnts"):
        self.image_paths = image_paths

        if type(self.image_paths[0]) == str:
            self.images = [cv2.imread(path) for path in image_paths]
        else:
            self.images = self.image_paths
            self.image_paths = ["00000.png", "00001.png"]
        self.copies = [img.copy() for img in self.images]
        self.points = [[] for _ in image_paths]
        self.show_point_indices = show_point_indices
        self.last_image_clicked = 0  # Index of the last image clicked
        self.prepare_images()

        self.save = save
        self.save_dir = save_dir if save_dir is not None else  osp.join(osp.dirname(osp.realpath(__file__)), "img_pnts")
        self.save_fs = save_fs
        self.end_fix = end_fix

        os.makedirs(self.save_dir, exist_ok=True)


    def pad_images_to_same_height(self):
        max_height = max(img.shape[0] for img in self.images)
        self.top_paddings = []
        for i, img in enumerate(self.images):
            diff = max_height - img.shape[0]
            self.images[i] = cv2.copyMakeBorder(img, diff // 2, diff - diff // 2, 0, 0, cv2.BORDER_CONSTANT)
            self.top_paddings.append(diff // 2)

    def prepare_images(self):
        self.pad_images_to_same_height()
        self.composite_image = np.hstack(self.images)
        self.original_composite_image = self.composite_image.copy()
        self.widths = [img.shape[1] for img in self.images]

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            scaled_x, scaled_y = x, y
            total_width = 0
            for idx, width in enumerate(self.widths):
                if scaled_x < total_width + width:
                    self.points[idx].append((scaled_x - total_width, scaled_y - self.top_paddings[idx]))
                    self.last_image_clicked = idx
                    print(f"Point added to Image {idx + 1}: ({scaled_x - total_width}, {scaled_y - self.top_paddings[idx]})")
                    break
                total_width += width
            self.draw_points()

    def remove_last_point(self):
        if self.points[self.last_image_clicked]:
            self.points[self.last_image_clicked].pop()
            print(f"Last point removed from Image {self.last_image_clicked + 1}")
            self.draw_points()

    def draw_points(self, show_img = True):
        self.composite_image = self.original_composite_image.copy()
        total_width = 0
        for idx, points in enumerate(self.points):
            for pidx, point in enumerate(points):
                pos0, pos1 = int(point[0]), int(point[1])
                cv2.circle(self.composite_image, (pos0 + total_width, pos1 + self.top_paddings[idx]), 2, (0, 255, 0), -1)
                if self.show_point_indices:
                    cv2.putText(self.composite_image, str(pidx), (pos0 + total_width + 10, pos1 + self.top_paddings[idx] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            total_width += self.widths[idx]
        
        if show_img:
            cv2.imshow('Composite Image', self.composite_image)

    def select_points(self):
        cv2.namedWindow('Composite Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Composite Image', self.original_composite_image.shape[1], self.original_composite_image.shape[0])
        cv2.setMouseCallback('Composite Image', self.click_event)

        while True:
            self.draw_points()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                self.remove_last_point()

        cv2.destroyAllWindows()

        if self.save:
            self.save_all_points()

        return np.array(self.points)

    def save_all_points(self):
        for i, (img_f, img_pnt) in enumerate(zip(self.image_paths, self.points)):
            if self.save_fs is not None:
                save_f = self.save_fs[i]
            else:
                save_f = osp.join(self.save_dir, osp.basename(img_f).split(".")[0] + f"_{self.end_fix}.npy")
            np.save(save_f, img_pnt)
    
    def select_checker(self):
        pnts_2d = np.stack([detect_chessboard(img).squeeze() for img in  self.images])
        self.points = pnts_2d
        # self.draw_points()

        if self.save:
            self.save_all_points()
        
        return pnts_2d

    def save_ref_img(self):
        ref_f = osp.join(self.save_dir, "ref_img.png")
        cv2.imwrite(ref_f, self.composite_image)
        print(ref_f)

if __name__ == "__main__":
    ##### test
    # Example usage
    img_f1 = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/halloween_b2_v1/halloween_b2_v1_recon/images/00000.png"
    img_f2 = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/halloween_b2_v1/trig_eimgs/0000.png"
    save_dir = osp.join(osp.dirname(osp.realpath(__file__)), "dev_img_pnts")

    selector = ImagePointSelector([img_f1, img_f2], show_point_indices=True, save_dir=save_dir)
    points_image1, points_image2 = selector.select_points()

    print("Points on Image 1:", points_image1)
    print("Points on Image 2:", points_image2)