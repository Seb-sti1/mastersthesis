"""
Try to create a BEV image using the ZED camera. Currently, it uses the images and camera intrinsics matrix.
"""
import argparse
import csv
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from numba import njit, prange
from numpy import ndarray
from tqdm import tqdm

from scripts.bev.rgbd_processing import create_pointcloud
from scripts.rosutils.rosbag import msg2image, msg2calibration, replay_bag
from scripts.utils.datasets import get_dataset_by_name
from scripts.utils.plot import DynamicO3DWindow


def keep_internal_disk(points: np.ndarray, colors: np.ndarray,
                       radius: float,
                       center: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if center is not None:
        points = points - center
    valid = np.linalg.norm(points, axis=1) <= radius
    return points[valid], colors[valid]


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
    # FIXME there might an inversion of left-right axis
    coordinates = (np.stack([(points[:, 2] - points[:, 2].min()),
                             (points[:, 0] - points[:, 0].min())], axis=-1) / bev_res).astype(np.uint16)
    n, m = coordinates[:, 0].max() + 1, coordinates[:, 1].max() + 1
    bev = compute_bev(coordinates, colors, n, m)
    return bev


def icp_and_merge(target: o3d.geometry.PointCloud,
                  points: ndarray, colors: ndarray,
                  trans_init: Optional[ndarray] = None,
                  icp_threshold: float = 0.02) -> Tuple[ndarray, ndarray]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if trans_init is None:
        trans_init = np.eye(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd, target, icp_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    pcd += target.transform(np.linalg.inv(reg_p2p.transformation))
    return pcd


def main(ros_bag: Path, benchmark: bool):
    rgb_t = "/zed2/zed_node/rgb/image_rect_color"
    depth_t = "/zed2/zed_node/depth/depth_registered"
    k_matrix_t = "/zed2/zed_node/rgb/camera_info"
    bev_res = 0.05  # meters
    bev_radius = 5  # meters construct the bev around the robot

    bench_file = open(f"bev_timings_{int(time.time())}.csv", "w") if benchmark else None
    bench_csv = csv.writer(bench_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL) if benchmark else None

    pcd = None

    pcd_gui = DynamicO3DWindow()
    is_running = False
    progress = tqdm(replay_bag(ros_bag, {rgb_t: msg2image,
                                         depth_t: msg2image,
                                         k_matrix_t: msg2calibration}))
    t_tot = 0
    for data in progress:
        rgb = data[rgb_t][:, :, :3] / 255.0  # rbga 0-255
        depth = data[depth_t]  # float32 meters
        k_matrix = data[k_matrix_t]

        t_pcd = - time.time_ns()
        points, colors = create_pointcloud(rgb, depth, k_matrix)
        t_pcd += time.time_ns()
        progress.set_description(f"pcd creation {t_pcd * 1e-9:>10.3f}s {points.shape[0]}pts")

        t_accumulate = - time.time_ns()
        # reduce to a circle
        points, colors = keep_internal_disk(points, colors, bev_radius)
        # icp
        if pcd is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd = icp_and_merge(pcd, points, colors)
        # reduce to a circle
        p, c = keep_internal_disk(np.asarray(pcd.points), np.asarray(pcd.colors), bev_radius)
        # sample down
        pcd.points = o3d.utility.Vector3dVector(p)
        pcd.colors = o3d.utility.Vector3dVector(c)
        pcd = pcd.voxel_down_sample(voxel_size=bev_res / 2)
        t_accumulate += time.time_ns()
        pcd_gui.show_pcd(pcd)
        progress.set_description(f"pcd creation {t_pcd * 1e-9:>10.3f}s {points.shape[0]}pts   "
                                 f"accumulate {t_accumulate * 1e-9:>10.3f}s {np.asarray(pcd.points).shape[0]}pts")

        t_bev = - time.time_ns()
        bev = create_bev(np.asarray(pcd.points), np.asarray(pcd.colors), bev_res)
        t_bev += time.time_ns()
        progress.set_description(f"pcd creation {t_pcd * 1e-9:>10.3f}s {points.shape[0]}pts   "
                                 f"accumulate {t_accumulate * 1e-9:>10.3f}s {np.asarray(pcd.points).shape[0]}pts   "
                                 f"bev {t_bev * 1e-9:>10.3f}s")

        if not benchmark:
            cv2.imshow("RGB", rgb)
            depth_image = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth_image = depth_image / depth_image.max()
            cv2.imshow("Depth", depth_image)
            cv2.imshow("BEV", bev)

            k = cv2.waitKey(0 if is_running else 20) & 0xFF
            if k == ord('q'):
                pcd_gui.finish()
                break
            elif k == ord(' '):
                is_running = not is_running
        else:
            t_tot += (t_pcd + t_bev) / 2
            bench_csv.writerow([t_pcd, t_bev, t_accumulate])

            if t_tot * 1e-9 > 20:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, default="rosbag_u2is/ENSTA_U2IS_grass_2024-05-03-14-54-33.bag")
    parser.add_argument("--bench", default=False, action="store_true")
    args = parser.parse_args()

    main(Path(get_dataset_by_name(args.bag)), args.bench)
