"""
This file is meant to contain utils function related to rosbag manipulations
"""

import argparse
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tqdm import tqdm

import rosbag
from scripts.utils import remove_special_characters


def extract_images_from_bag(bag_file: Path, topics: List[str], output_dir: Optional[Path] = None) -> None:
    """
    Extracts images from a bag file and saves them to the output directory
    :param bag_file: the bag file to extract from
    :param topics: the topics list
    :param output_dir: the output directory. By default, it creates a folder with the same name as the bag file.
    :return:
    """
    if output_dir is None:
        output_dir = bag_file.parent.joinpath(bag_file.stem)

    bag = rosbag.Bag(bag_file)
    bridge = CvBridge()
    count = 0
    for topic, msg, t in tqdm(bag.read_messages(topics=topics)):
        topic_dir = remove_special_characters(topic)
        if not output_dir.joinpath(topic_dir).exists():
            output_dir.joinpath(topic_dir).mkdir(parents=True)
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        save_path = str(output_dir.joinpath(topic_dir, f"frame_{count:06d}_{str(t.secs * int(1e9) + t.nsecs)}"))
        if cv_image.dtype in [np.float32, np.uint16]:
            np.save(save_path + ".npy", cv_image)
        else:
            cv2.imwrite(save_path + ".png", cv_image)
        count += 1
    bag.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract images from a ROS bag file')
    parser.add_argument('bag_file', type=Path, help='The bag file to extract from')
    parser.add_argument('topics', type=str, nargs='+', help='The topics to extract')
    parser.add_argument('--output_dir', type=str, help='The output directory', default=None)
    args = parser.parse_args()

    extract_images_from_bag(Path(args.bag_file), args.topics,
                            None if args.output_dir is None else Path(args.output_dir))
