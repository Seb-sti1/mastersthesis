from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rosutils.rosbag import replay_bag, msg2image, msg2gnss, withTimestamp, msg2alt, msg2gimbalStatus
from utils.datasets import get_dataset_by_name


def extract_barakuda(ros_bag: Path, output_path: Path):
    rgb_folder = (output_path / "ugv_rgb")
    depth_folder = (output_path / "ugv_depth")
    rgb_folder.mkdir(parents=True, exist_ok=True)
    depth_folder.mkdir(parents=True, exist_ok=True)

    gnss_t = "/imu/nav_sat_fix"
    depth_t = "/zed2i/zed_node/depth/depth_registered"
    rgb_t = "/zed2i/zed_node/rgb/image_rect_color"
    gnss_df = pd.DataFrame(columns=["timestamp", "latitude", "longitude", "altitude"])
    for data in replay_bag(ros_bag, {rgb_t: withTimestamp(msg2image),
                                     depth_t: withTimestamp(msg2image),
                                     gnss_t: withTimestamp(msg2gnss)},
                           exhaustive=True):

        if rgb_t in data:
            t, rgb = data[rgb_t]
            # for some reason shape is 360 640 3
            cv2.imwrite(str(rgb_folder / f"rgb_{t}.png"), rgb[:, :, :3])
        if depth_t in data:
            t, depth = data[depth_t]
            np.save(str(depth_folder / f"depth_{t}.npy"), depth)
        if gnss_t in data:
            t, gnss = data[gnss_t]
            gnss_df.loc[len(gnss_df)] = [int(str(t)), gnss["latitude"], gnss["longitude"], gnss["altitude"]]
    gnss_df.to_csv(output_path / "ugv_gnss.csv")


def extract_tundra(ros_bag: Path, output_path: Path):
    front_rgb_folder = (output_path / "uav_rgb_front")
    gimbal_rgb_folder = (output_path / "uav_rgb_gimbal")
    front_rgb_folder.mkdir(parents=True, exist_ok=True)
    gimbal_rgb_folder.mkdir(parents=True, exist_ok=True)

    gnss_t = "/mavros/global_position/raw/fix"
    alt_t = "/mavros/global_position/rel_alt"
    front_rgb_t = "/front/compressed"
    gimbal_rgb_t = "/gimbal/compressed"
    gimbal_status_t = "/viewpro/status"

    gnss_df = pd.DataFrame(columns=["timestamp", "latitude", "longitude", "altitude"])
    alt_df = pd.DataFrame(columns=["timestamp", "altitude"])
    gimbal_status_df = pd.DataFrame(columns=["timestamp", "roll", "pitch", "yaw", "manually set zoom", "zoom"])

    for data in replay_bag(ros_bag, {front_rgb_t: withTimestamp(msg2image),
                                     gimbal_rgb_t: withTimestamp(msg2image),
                                     gnss_t: withTimestamp(msg2gnss),
                                     alt_t: withTimestamp(msg2alt),
                                     gimbal_status_t: withTimestamp(msg2gimbalStatus)},
                           exhaustive=True):
        if front_rgb_t in data:
            t, rgb = data[front_rgb_t]
            cv2.imwrite(str(front_rgb_folder / f"front_rgb_{t}.png"), rgb)
        if gimbal_rgb_t in data:
            t, rgb = data[gimbal_rgb_t]
            cv2.imwrite(str(gimbal_rgb_folder / f"gimbal_rgb_{t}.png"), rgb)
        if gimbal_status_t in data:
            t, status = data[gimbal_status_t]
            gimbal_status_df.loc[len(gimbal_status_df)] = [int(str(t)), status["roll"], status["pitch"],
                                                           status["yaw"], status["manually set zoom"], status["zoom"]]
        if alt_t in data:
            t, alt = data[alt_t]
            alt_df.loc[len(alt_df)] = [int(str(t)), alt]
        if gnss_t in data:
            t, gnss = data[gnss_t]
            gnss_df.loc[len(gnss_df)] = [int(str(t)), gnss["latitude"], gnss["longitude"], gnss["altitude"]]
    gnss_df.to_csv(output_path / "uav_gnss.csv")
    gimbal_status_df.to_csv(output_path / "uav_gimbal_status.csv")


if __name__ == "__main__":
    extract_tundra(Path(get_dataset_by_name("rosbag_u2is/tundra_hub_drone_130625_1.bag")),
                   Path(get_dataset_by_name("rosbag_u2is")) / "hub_drone_130625")
    extract_barakuda(Path(get_dataset_by_name("rosbag_u2is/barakuda_hub_drone_130625_1.bag")),
                     Path(get_dataset_by_name("rosbag_u2is")) / "hub_drone_130625")
