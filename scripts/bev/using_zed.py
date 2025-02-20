"""
Try to create a BEV image using the ZED camera.

Currently, it uses rgb/image_rect_color and depth/depth_registered topics from the ZED camera.
"""
import cv2

from scripts.utils.datasets import get_dataset_by_name, load_files, select_png


def main():
    # Get the dataset. It is extracted from a rosbag file with the same name using script/rosutils/rosbag.py
    rgb_path = get_dataset_by_name("rosbag_husky/ENSTA_U2IS_grass_2024-05-03-14-54-33/zed2zednodergbimagerectcolor")
    depth_path = get_dataset_by_name(
        "rosbag_husky/ENSTA_U2IS_grass_2024-05-03-14-54-33/zed2zednodedepthdepthregistered")

    for rgb, depth in zip(load_files(rgb_path), load_files(depth_path)):
        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", depth)
        print(f"RGB shape: {rgb.shape}, Depth shape: {depth.shape}, {depth[depth != 0].min()}, {depth.max()}")
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
