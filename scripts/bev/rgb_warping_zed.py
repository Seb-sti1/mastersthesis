import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from scripts.bev.rgbd_processing import create_matrix_coordinate, change_frame_matrix_coordinate
from scripts.rosutils.rosbag import replay_bag, msg2image, msg2calibration, msg2imu
from scripts.utils.datasets import get_dataset_by_name, resize_image


def reject_too_high_points(rgb: np.ndarray, depth: np.ndarray,
                           k_matrix: np.ndarray,
                           tf_camera_base_link: np.ndarray,
                           max_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Will project the points in the frame of the robot to exclude structure that are too high.

    Hypothesis:
      - Ground locally flat
      - The robot tf (base_link) XY plane is parallel to the ground

    :param rgb: the rgb image
    :param depth: the depth image
    :param k_matrix: the camera intrinsics matrix
    :param tf_camera_base_link: the tf that transform a point in the camera frame to the horizontal of the robot
    :param max_height: the maximum height from the ground that is valid
    :return: the rgb image where the pixels above max_height are removed
    """
    xyz = create_matrix_coordinate(depth, k_matrix)
    h, w, _ = xyz.shape
    valid = np.isfinite(xyz[:, :, 0])

    xyz_in_base_link = change_frame_matrix_coordinate(xyz, tf_camera_base_link)
    # TODO improve xyz_in_base_link < max_height by actually detecting the ground
    valid_h = valid & (xyz_in_base_link[:, :, 2] < max_height)

    # Use to check the point cloud
    # d = DynamicO3DWindow()
    # points = xyz_in_base_link[valid_h]
    # colors = rgb[valid_h]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # d.show_pcd(pcd)
    # d.finish()

    new_rgb = rgb.copy()
    new_rgb[np.invert(valid_h)] = [0, 0, 0]
    return new_rgb, xyz


def from_3d_to_pixel(point: np.ndarray, K, P):
    pass


def create_bev(rgb: np.ndarray, tf_camera_horizontal: np.ndarray):
    """

    :param rgb:
    :param tf_horizontal_base_link: the rotation (no translation) from a horizontal frame (z along the gravity) to the base_link of the robot

    :return:
    """
    pass


def main(ros_bag: Path, benchmark: bool):
    rgb_t = "/zed2i/zed_node/rgb/image_rect_color"
    depth_t = "/zed2i/zed_node/depth/depth_registered"
    k_matrix_t = "/zed2i/zed_node/rgb/camera_info"
    imu_t = "/zed2i/zed_node/imu/data"

    # TODO actually take the right tf
    tf_camera_base_link = np.array([[0., 0., 1., 0.],
                                    [-1., 0., 0., 0.],
                                    [0., -1, 0., 0.],
                                    [0., 0, 0., 1.]])

    # d = DynamicO3DWindow()
    is_running = False
    progress = tqdm(replay_bag(ros_bag, {rgb_t: msg2image,
                                         depth_t: msg2image,
                                         k_matrix_t: msg2calibration,
                                         imu_t: msg2imu}))
    for data in progress:
        rgb = data[rgb_t][:, :, :3] / 255.0  # rbga 0-255
        depth = data[depth_t]  # float32 meters
        K = data[k_matrix_t]
        imu = data[imu_t]

        rgb_filtered, xyz = reject_too_high_points(rgb, depth, K,
                                                   tf_camera_base_link,
                                                   0)

        # R = o3d.geometry.get_rotation_matrix_from_quaternion(
        #     [imu['orientation'][0], *imu['orientation'][:3]])
        # euler_angles = Rotation.from_quat(imu['orientation']).as_euler('zyx')
        R_top_down = Rotation.from_euler('zyx', [0, 0, np.pi / 5]).as_matrix()
        # R = tf_camera_base_link[:3, :3]
        # M = np.matmul(K, np.matmul(R, np.linalg.inv(K)))
        # R_top_down = np.array([[1, 0, 0],
        #                        [0, 0, -1],  # Aligns the Z-axis to point down
        #                        [0, 1, 0]], dtype=np.float32)

        M = np.matmul(K, np.matmul(tf_camera_base_link[:3, :3], np.linalg.inv(K)))

        # compute appropriate shift and size so that the reprojected image fit in the matrix
        corners = np.matmul(M, np.array([[0, 0, 1], [0, rgb.shape[0], 1],
                                         [rgb.shape[1], 0, 1], [rgb.shape[1], rgb.shape[0], 1]]).T)
        corners /= corners[2, :]
        print(corners)
        min_x, min_y = np.min(corners[:2], axis=1)
        max_x, max_y = np.max(corners[:2], axis=1)
        T = np.eye(3)
        T[0, 2] = -min_x  # Shift right
        T[1, 2] = -min_y  # Shift down
        M = np.matmul(T, M)  # Apply translation

        new_width = int(np.ceil(max_x - min_x))
        new_height = int(np.ceil(max_y - min_y))
        rgb_warp = cv2.warpPerspective(rgb, M, (new_width, new_height))
        rgb_filtered_warp = cv2.warpPerspective(rgb_filtered, M, (new_width, new_height))

        # valid = np.isfinite(xyz_in_base_link[:, :, 0])
        #
        # tf_base_link_horizontal = np.eye(4, dtype=np.float32)
        # tf_base_link_horizontal[:3, :3] = np.linalg.inv(o3d.geometry.get_rotation_matrix_from_quaternion(
        #     [imu['orientation'][0], *imu['orientation'][:3]]))
        # xyz_in_world = change_frame_matrix_coordinate(xyz_in_base_link, tf_base_link_horizontal)
        #
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz_in_world[valid])
        # Use to check the point cloud
        # plane_model, inliers = pcd.segment_plane(distance_threshold=0.25,
        #                                          ransac_n=3,
        #                                          num_iterations=1000)
        #
        # inlier_cloud = pcd.select_by_index(inliers)
        # inlier_cloud.paint_uniform_color([0., 1., 0])
        # outlier_cloud = pcd.select_by_index(inliers, invert=True)
        # outlier_cloud.paint_uniform_color([1.0, 0, 0])
        # d.show_pcd(inlier_cloud + outlier_cloud)

        if not benchmark:
            cv2.imshow("RGB", rgb)
            cv2.imshow("RGB filtered", rgb_filtered)
            cv2.imshow("Warp RGB filtered", resize_image(rgb_filtered_warp, 1000, 1000))
            cv2.imshow("Warp RGB", resize_image(rgb_warp, 1000, 1000))

            k = cv2.waitKey(20 if is_running else 0) & 0xFF
            if k == ord('q'):
                d.finish()
                break
            elif k == ord(' '):
                is_running = not is_running
        else:
            pass
            # bench_csv.writerow([t_pcd, t_bev, t_accumulate])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str,
                        default="rosbag_u2is/pal_diff_ang_2024-04-18-14-07-54.bag")
    parser.add_argument("--bench", default=False, action="store_true")
    args = parser.parse_args()

    main(Path(get_dataset_by_name(args.bag)), args.bench)
