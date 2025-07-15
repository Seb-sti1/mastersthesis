"""
Create a bev image using a homography computed by finding the real world coordinates of the corners of bottom half of the
ugv images
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from rosutils.rosbag import withTimestamp
from scripts.rosutils.rosbag import replay_bag, msg2image, msg2calibration
from scripts.utils.datasets import get_dataset_by_name, resize_image


def main(ros_bag: Path, benchmark: bool, output_path: Path):
    rgb_t = "/zed2i/zed_node/rgb/image_rect_color"
    k_matrix_t = "/zed2i/zed_node/rgb/camera_info"
    output_path.mkdir(exist_ok=True, parents=True)
    # rosrun tf tf_echo base_link zed2i_base_link
    # - Translation: [0.966, -0.000, 0.694]
    # - Rotation: in Quaternion [0.000, 0.137, 0.000, 0.991]
    #             in RPY (radian) [0.000, 0.275, 0.000]
    #             in RPY (degree) [0.000, 15.750, 0.000]
    # rosrun tf tf_echo foot_print_link zed2i_right_camera_optical_frame
    # - Translation: [0.961, -0.060, 1.174]
    # - Rotation: in Quaternion [0.564, -0.564, 0.427, -0.427]
    #             in RPY (radian) [-1.846, -0.000, -1.571]
    #             in RPY (degree) [-105.750, -0.000, -90.000]
    # root@6ae77e38a582:/catkin_ws# rosrun tf tf_echo foot_print_link zed2i_left_camera_optical_frame
    # - Translation: [0.961, 0.060, 1.174]
    # - Rotation: in Quaternion [0.564, -0.564, 0.427, -0.427]
    #             in RPY (radian) [-1.846, -0.000, -1.571]
    #             in RPY (degree) [-105.750, -0.000, -90.000]

    R = Rotation.from_quat([0.564, -0.564, 0.427, -0.427])
    tf_camera2robot = np.eye(4)
    tf_camera2robot[:3, :3] = R.as_matrix()
    tf_camera2robot[:3, 3] = [0.961, 0, 1.174]

    is_running = False
    progress = tqdm(replay_bag(ros_bag, {rgb_t: withTimestamp(msg2image),
                                         k_matrix_t: msg2calibration}))
    for data in progress:
        timestamp, rgb = data[rgb_t]
        rgb = rgb[:, :, :3] / 255.0  # rbga 0-255
        K = data[k_matrix_t]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        ref_points = np.array([[0, rgb.shape[0]],
                               [rgb.shape[1], rgb.shape[0]],
                               [rgb.shape[1], rgb.shape[0] // 2],
                               [0, rgb.shape[0] // 2]])
        corners = np.zeros((4, 2))
        for idx, (j, i) in enumerate(ref_points):
            z_ground_in_camera = - tf_camera2robot[2, 3] / (tf_camera2robot[2, 0] * (j - cx) / fx +
                                                            tf_camera2robot[2, 1] * (i - cy) / fy +
                                                            tf_camera2robot[2, 2])
            x_ground_in_camera = (j - cx) / fx * z_ground_in_camera
            y_ground_in_camera = (i - cy) / fy * z_ground_in_camera
            x_ground_in_base_footprint = np.matmul(tf_camera2robot, np.array([[x_ground_in_camera,
                                                                               y_ground_in_camera,
                                                                               z_ground_in_camera,
                                                                               1]]).T)[:3, 0]
            corners[idx, :] = x_ground_in_base_footprint[:2]
        corners = - np.flip(corners * 200, axis=1)
        corners -= corners[3, :]
        M, mask = cv2.findHomography(ref_points, corners)
        rgb_warp = cv2.warpPerspective(rgb, M, np.max(corners, axis=0).astype(np.int32))
        if not benchmark:
            cv2.imshow("RGB", rgb)
            cv2.imshow("Warp RGB", resize_image(rgb_warp, 1000, 1000))

            k = cv2.waitKey(20 if is_running else 0) & 0xFF
            if k == ord('q'):
                # d.finish()
                break
            elif k == ord(' '):
                is_running = not is_running
        else:
            cv2.imwrite(str(output_path / f"bev_{timestamp}.png"), (255 * rgb_warp).astype(np.int32))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str,
                        default="rosbag_u2is/barakuda_hub_drone_130625_1.bag")
    parser.add_argument("--bench", default=True, action="store_true")
    args = parser.parse_args()

    main(Path(get_dataset_by_name(args.bag)), args.bench,
         Path(get_dataset_by_name("rosbag_u2is")) / "hub_drone_130625" / "ugv_rgb_bev")
