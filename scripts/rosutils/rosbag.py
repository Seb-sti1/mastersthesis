"""
This file is for rosbag manipulations
"""

from pathlib import Path
from typing import Callable, Dict, Any, Iterator

import cv2
import numpy as np
from cv_bridge import CvBridge
from tqdm import tqdm

import rosbag

bridge = CvBridge()


def withTimestamp(func):
    return lambda msg, t: (t, func(msg, t))


def msg2gnss(msg, t):
    return {
        "latitude": msg.latitude,
        "longitude": msg.longitude,
        "altitude": msg.altitude
    }


def msg2alt(msg, t):
    return msg


def msg2gimbalStatus(msg, t):
    return {
        "roll": msg.data[0],
        "pitch": msg.data[1],
        "yaw": msg.data[2],
        "manually set zoom": msg.data[3],
        "zoom": msg.data[4],
    }


def msg2imu(msg, t):
    return {
        "orientation": np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]),
        "orientation_covariance": np.array(msg.orientation_covariance).reshape(3, 3),
        "angular_velocity": np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]),
        "angular_velocity_covariance": np.array(msg.angular_velocity_covariance).reshape(3, 3),
        "linear_acceleration": np.array(
            [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]),
        "linear_acceleration_covariance": np.array(msg.linear_acceleration_covariance).reshape(3, 3),
    }


def msg2image(msg, t):
    if "CompressedImage" in str(type(msg)):
        np_arr = np.fromstring(msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")


def msg2calibration(msg, t):
    return np.reshape(msg.K, (3, 3))


def replay_bag(bag_file: Path, topics: Dict[str, Callable], exhaustive: bool = False) -> Iterator[Dict[str, Any]]:
    bag = rosbag.Bag(bag_file)
    found = {t: False for t in topics.keys()}
    last_published = {t: (None, None) for t in topics.keys()}

    for topic, msg, t in tqdm(bag.read_messages(topics=topics), leave=False):
        last_published[topic] = (msg, t)
        found[topic] = True

        if (exhaustive and any(found.values())) or (not exhaustive and all(found.values())):
            yield {name: topics[name](msg, t) for name, (msg, t) in last_published.items() if msg is not None}
            found = {t: False for t in topics.keys()}
            if exhaustive:
                # when it is exhaustive replay there is a need to reset last_published
                # as otherwise the data will be yielded multiple time
                last_published = {t: (None, None) for t in topics.keys()}

    bag.close()
    return None
