"""
Try to create a BEV image using the ZED camera. Currently, it uses the images and camera intrinsics matrix.
"""
from pathlib import Path

import cv2
import numpy as np

from scripts.rosutils.rosbag import msg2image, msg2calibration, replay_bag
from scripts.utils.datasets import get_dataset_by_name


def main(ros_bag: Path):
    rgb_t = "/zed2/zed_node/rgb/image_rect_color"
    depth_t = "/zed2/zed_node/depth/depth_registered"
    k_matrix_t = "/zed2/zed_node/rgb/camera_info"

    for data in replay_bag(ros_bag, {rgb_t: msg2image,
                                     depth_t: msg2image,
                                     k_matrix_t: msg2calibration}):
        rgb = data[rgb_t]
        depth = data[depth_t]
        k_matrix = data[k_matrix_t]

        cv2.imshow("RGB", rgb)
        depth_image = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_image = depth_image / depth_image.max()
        cv2.imshow("Depth", depth_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main(Path(get_dataset_by_name("rosbag_husky/ENSTA_U2IS_grass_2024-05-03-14-54-33.bag")))
