from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rosutils.rosbag import replay_bag, msg2image, msg2gnss, withTimestamp, msg2alt, msg2gimbalStatus, msg2imu, toEuler
from utils.datasets import get_dataset_by_name


def extract_barakuda(ros_bag: Path, output_path: Path):
    rgb_folder = (output_path / "ugv_rgb")
    depth_folder = (output_path / "ugv_depth")
    rgb_folder.mkdir(parents=True, exist_ok=True)
    depth_folder.mkdir(parents=True, exist_ok=True)

    gnss_t = "/imu/nav_sat_fix"
    sbg_imu_t = "/imu/data"
    zed_imu_t = "/zed2i/zed_node/imu/data"
    depth_t = "/zed2i/zed_node/depth/depth_registered"
    rgb_t = "/zed2i/zed_node/rgb/image_rect_color"
    gnss_df = pd.DataFrame(columns=["timestamp", "latitude", "longitude", "altitude"])
    sbg_euler_df = pd.DataFrame(columns=["timestamp", "roll", "pitch", "yaw"])
    zed_euler_df = pd.DataFrame(columns=["timestamp", "roll", "pitch", "yaw"])

    for data in replay_bag(ros_bag, {rgb_t: withTimestamp(msg2image),
                                     sbg_imu_t: withTimestamp(toEuler(msg2imu)),
                                     zed_imu_t: withTimestamp(toEuler(msg2imu)),
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
        if sbg_imu_t in data:
            t, euler = data[sbg_imu_t]
            sbg_euler_df.loc[len(sbg_euler_df)] = [int(str(t)), euler[0], euler[1], euler[2]]
        if zed_imu_t in data:
            t, euler = data[zed_imu_t]
            zed_euler_df.loc[len(zed_euler_df)] = [int(str(t)), euler[0], euler[1], euler[2]]
    gnss_df.to_csv(output_path / "ugv_gnss.csv")
    sbg_euler_df.to_csv(output_path / "ugv_sbg_euler.csv")
    zed_euler_df.to_csv(output_path / "ugv_zed_euler.csv")

    gnss_extended_df = pd.DataFrame(columns=["timestamp", "longitude", "latitude", "altitude",
                                             "sbg_roll", "sbg_pitch", "sbg_yaw",
                                             "zed_roll", "zed_pitch", "zed_yaw",
                                             "delta_sbg", "delta_zed"])

    def get_closest(df, date):
        return df.iloc[(df['timestamp'] - date).abs().argmin()]

    for idx, row in gnss_df.iterrows():
        sbg_euler = get_closest(sbg_euler_df, row["timestamp"])
        zed_euler = get_closest(zed_euler_df, row["timestamp"])

        gnss_extended_df.loc[idx] = [row["timestamp"], row["longitude"], row["latitude"], row["altitude"],
                                     sbg_euler["roll"], sbg_euler["pitch"], sbg_euler["yaw"],
                                     zed_euler["roll"], zed_euler["pitch"], zed_euler["yaw"],
                                     (row["timestamp"] - sbg_euler["timestamp"]) * 1e-9,
                                     (row["timestamp"] - zed_euler["timestamp"]) * 1e-9]

    n = 15
    longitudes = gnss_extended_df["longitude"].to_numpy()
    latitudes = gnss_extended_df["latitude"].to_numpy()
    diff_longitudes = longitudes[:-n] - longitudes[n:]
    diff_latitudes = latitudes[:-n] - latitudes[n:]
    yaw = np.arctan2(diff_latitudes, diff_longitudes)
    valid = (np.absolute(diff_longitudes) >= 1e-6) & (np.absolute(diff_latitudes) >= 1e-6)
    yaw_v = np.zeros_like(longitudes)
    last = 0
    for idx in range(len(yaw_v)):
        if idx < len(diff_longitudes) and valid[idx]:
            last = yaw[idx]
        yaw_v[idx] = last
    gnss_extended_df["gps_heading"] = yaw_v + np.pi

    gnss_extended_df.to_csv(output_path / "ugv_gnss_extended.csv")


def extract_tundra(ros_bag: Path, output_path: Path):
    front_rgb_folder = (output_path / "uav_rgb_front")
    gimbal_rgb_folder = (output_path / "uav_rgb_gimbal")
    front_rgb_folder.mkdir(parents=True, exist_ok=True)
    gimbal_rgb_folder.mkdir(parents=True, exist_ok=True)

    gnss_t = "/mavros/global_position/raw/fix"
    imu_t = "/mavros/imu/data"
    alt_t = "/mavros/global_position/rel_alt"
    front_rgb_t = "/front/compressed"
    gimbal_rgb_t = "/gimbal/compressed"
    gimbal_status_t = "/viewpro/status"

    gnss_df = pd.DataFrame(columns=["timestamp", "latitude", "longitude", "altitude"])
    alt_df = pd.DataFrame(columns=["timestamp", "altitude"])
    gimbal_status_df = pd.DataFrame(columns=["timestamp", "roll", "pitch", "yaw", "manually_set_zoom", "zoom"])
    euler_df = pd.DataFrame(columns=["timestamp", "roll", "pitch", "yaw"])

    for data in replay_bag(ros_bag, {
        front_rgb_t: withTimestamp(msg2image),
        gimbal_rgb_t: withTimestamp(msg2image),
        imu_t: withTimestamp(toEuler(msg2imu)),
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
        if imu_t in data:
            t, euler = data[imu_t]
            euler_df.loc[len(euler_df)] = [int(str(t)), euler[0], euler[1], euler[2]]
        if alt_t in data:
            t, alt = data[alt_t]
            alt_df.loc[len(alt_df)] = [int(str(t)), alt]
        if gnss_t in data:
            t, gnss = data[gnss_t]
            gnss_df.loc[len(gnss_df)] = [int(str(t)), gnss["latitude"], gnss["longitude"], gnss["altitude"]]
    gnss_df.to_csv(output_path / "uav_gnss.csv")
    alt_df.to_csv(output_path / "uav_alt.csv")
    euler_df.to_csv(output_path / "uav_euler.csv")
    gimbal_status_df.to_csv(output_path / "uav_gimbal_status.csv")

    gnss_extended_df = pd.DataFrame(columns=["timestamp", "longitude", "latitude", "altitude_bad",
                                             "altitude",
                                             "camera_roll", "camera_pitch", "camera_yaw",
                                             "camera_manually_set_zoom", "camera_zoom",
                                             "roll", "pitch", "yaw",
                                             "delta_alt",
                                             "delta_gimbal",
                                             "delta_euler"])

    def get_closest(df, date):
        return df.iloc[(df['timestamp'] - date).abs().argmin()]

    for idx, row in gnss_df.iterrows():
        alt = get_closest(alt_df, row["timestamp"])
        gimbal = get_closest(gimbal_status_df, row["timestamp"])
        euler = get_closest(euler_df, row["timestamp"])

        gnss_extended_df.loc[idx] = [row["timestamp"], row["longitude"], row["latitude"], row["altitude"],
                                     alt["altitude"],
                                     gimbal["roll"], gimbal["pitch"], gimbal["yaw"],
                                     gimbal["manually_set_zoom"], gimbal["zoom"],
                                     euler["roll"], euler["pitch"], euler["yaw"],
                                     (row["timestamp"] - alt["timestamp"]) * 1e-9,
                                     (row["timestamp"] - gimbal["timestamp"]) * 1e-9,
                                     (row["timestamp"] - euler["timestamp"]) * 1e-9]

    gnss_extended_df.to_csv(output_path / "uav_gnss_extended.csv")


if __name__ == "__main__":
    extract_tundra(Path(get_dataset_by_name("rosbag_u2is/tundra_hub_drone_130625_1.bag")),
                   Path(get_dataset_by_name("rosbag_u2is")) / "hub_drone_130625")
    extract_barakuda(Path(get_dataset_by_name("rosbag_u2is/barakuda_hub_drone_130625_1.bag")),
                     Path(get_dataset_by_name("rosbag_u2is")) / "hub_drone_130625")
