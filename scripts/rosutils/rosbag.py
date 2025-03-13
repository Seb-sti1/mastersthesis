"""
This file is for rosbag manipulations
"""

from pathlib import Path
from typing import Callable, Dict, Any, Iterator

import numpy as np
from cv_bridge import CvBridge
from tqdm import tqdm

import rosbag

bridge = CvBridge()


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
    return bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")


def msg2calibration(msg, t):
    return np.reshape(msg.K, (3, 3))


def replay_bag(bag_file: Path, topics: Dict[str, Callable]) -> Iterator[Dict[str, Any]]:
    bag = rosbag.Bag(bag_file)
    found = {t: False for t in topics.keys()}
    last_published = {t: (None, None) for t in topics.keys()}

    for topic, msg, t in tqdm(bag.read_messages(topics=topics), leave=False):
        last_published[topic] = (msg, t)
        found[topic] = True

        if all(found.values()):
            yield {name: topics[name](msg, t) for name, (msg, t) in last_published.items()}
            found = {t: False for t in topics.keys()}
    bag.close()
    return None
