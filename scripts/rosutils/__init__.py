import os


def this_needs_ros():
    if not os.getenv("ROS_DISTRO"):
        raise ImportError("This module requires ROS python package.\n"
                          "Please see more information in scripts/README.md file.")


this_needs_ros()
