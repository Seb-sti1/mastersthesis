"""
Try to create a BEV image using the ZED camera. Currently, it uses the images and camera intrinsics matrix.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from numba import njit, prange
from tqdm import tqdm

from scripts.rosutils.rosbag import msg2image, msg2calibration, replay_bag
from scripts.utils.datasets import get_dataset_by_name


def pcd_vis(points, colors, pcd=None):
    if pcd is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Create visualizer and set background to black
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def create_pointcloud(rgb, depth, k_matrix):
    """
    Create a point cloud from the rgb image and the depth image.

    The return point coordinates are as follows:
     - points[:, 0] left right
     - points[:, 1] top down
     - points[:, 2] front back

    :param rgb: the rgb image
    :param depth: the depth image (float32 meters)
    :param k_matrix: the camera intrinsics matrix
    :return: points, colors
    """
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

    # TODO robot inclination into consideration

    return points, colors


@njit(parallel=True)
def compute_bev(coordinates, colors, n, m):
    bev = np.zeros((n, m, 3))
    for i in prange(n):
        for j in prange(m):
            in_range = (coordinates[:, 0] == i) & (coordinates[:, 1] == j)
            c = np.count_nonzero(in_range)
            if c > 0:
                bev[i, j, :] = np.sum(colors[in_range, :], axis=0) / c
    return bev


def create_bev(points: np.ndarray, colors: np.ndarray, bev_res: float) -> np.ndarray:
    coordinates = (np.stack([(points[:, 2] - points[:, 2].min()),
                             (points[:, 0] - points[:, 0].min())], axis=-1) / bev_res).astype(np.uint16)
    n, m = coordinates[:, 0].max() + 1, coordinates[:, 1].max() + 1
    bev = compute_bev(coordinates, colors, n, m) / 255.0
    return bev


def main(ros_bag: Path, benchmark: bool):
    rgb_t = "/zed2/zed_node/rgb/image_rect_color"
    depth_t = "/zed2/zed_node/depth/depth_registered"
    k_matrix_t = "/zed2/zed_node/rgb/camera_info"
    bev_res = 0.05  # meters

    bench_file = open(f"bev_timings_{int(time.time())}.csv", "w") if benchmark else None
    bench_csv = csv.writer(bench_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL) if benchmark else None

    t_tot = 0
    for data in tqdm(replay_bag(ros_bag, {rgb_t: msg2image,
                                          depth_t: msg2image,
                                          k_matrix_t: msg2calibration})):
        rgb = data[rgb_t][:, :, :3]  # rbga 0-255
        depth = data[depth_t]  # float32 meters
        k_matrix = data[k_matrix_t]

        t_pcd = - time.time_ns()
        points, colors = create_pointcloud(rgb, depth, k_matrix)
        t_pcd += time.time_ns()

        t_bev = - time.time_ns()
        bev = create_bev(points, colors, bev_res)
        t_bev += time.time_ns()

        if not benchmark:
            cv2.imshow("RGB", rgb)
            depth_image = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth_image = depth_image / depth_image.max()
            cv2.imshow("Depth", depth_image)
            cv2.imshow("BEV", bev)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            t_tot += (t_pcd + t_bev) / 2
            bench_csv.writerow([t_pcd, t_bev])

            if t_tot * 1e-9 > 20:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, default="rosbag_husky/ENSTA_U2IS_grass_2024-05-03-14-54-33.bag")
    parser.add_argument("--bench", default=False, action="store_true")
    args = parser.parse_args()

    main(Path(get_dataset_by_name(args.bag)), args.bench)
