"""
Try to create a BEV image using the ZED camera. Currently, it uses the images and camera intrinsics matrix.
"""
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from scripts.rosutils.rosbag import msg2image, msg2calibration, replay_bag
from scripts.utils.datasets import get_dataset_by_name


def create_pointcloud(rgb, depth, k_matrix):
    height, width = depth.shape
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    valid = np.isfinite(z) & (z > 0)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x[valid], y[valid], z[valid]), axis=-1)
    colors = rgb[valid]

    return points, colors


def pcd_vis(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Create visualizer and set background to black
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # vis.get_render_option().background_color = np.asarray([0, 0, 0])  # black background
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def naive_bev(points: np.ndarray, colors: np.ndarray, bev_res: float) -> np.ndarray:
    coordinates = (np.stack([(points[:, 2] - points[:, 2].min()),
                             (points[:, 0] - points[:, 0].min())], axis=-1) / bev_res).astype(np.uint16)
    n, m = coordinates[:, 0].max() + 1, coordinates[:, 1].max() + 1
    bev = np.zeros((n, m, 3))

    for i in range(n):
        for j in range(m):
            in_range = (coordinates[:, 0] == i) & (coordinates[:, 1] == j)
            if np.count_nonzero(in_range) > 0:
                bev[i, j] = np.average(colors[in_range, :], axis=0)

    return bev / 255.0


def main(ros_bag: Path):
    rgb_t = "/zed2/zed_node/rgb/image_rect_color"
    depth_t = "/zed2/zed_node/depth/depth_registered"
    k_matrix_t = "/zed2/zed_node/rgb/camera_info"
    bev_res = 0.1  # meters

    for data in replay_bag(ros_bag, {rgb_t: msg2image,
                                     depth_t: msg2image,
                                     k_matrix_t: msg2calibration}):
        rgb = data[rgb_t][:, :, :3]
        depth = data[depth_t]
        k_matrix = data[k_matrix_t]

        points, colors = create_pointcloud(rgb, depth, k_matrix)
        # points[:, 0] left right
        # points[:, 1] top down
        # points[:, 2] front back
        bev = naive_bev(points, colors, bev_res)

        cv2.imshow("RGB", rgb)
        depth_image = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_image = depth_image / depth_image.max()
        cv2.imshow("Depth", depth_image)
        cv2.imshow("BEV", bev)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main(Path(get_dataset_by_name("rosbag_husky/ENSTA_U2IS_grass_2024-05-03-14-54-33.bag")))
