from typing import Tuple

import numpy as np


def create_matrix_coordinate(z: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    """
    :param z: the depth image (float32 meters)
    :param k_matrix: the camera intrinsics matrix
    :return: An image of the coordinates of each pixel in the camera frame
    """
    height, width = z.shape
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    xyz = np.stack([u, v, z], axis=-1)

    xyz[:, :, 0] = (xyz[:, :, 0] - cx) * xyz[:, :, 2] / fx
    xyz[:, :, 1] = (xyz[:, :, 1] - cy) * xyz[:, :, 2] / fy

    valid = np.invert(np.isfinite(z) & (z > 0))
    xyz[valid, 0] = np.nan
    xyz[valid, 1] = np.nan
    xyz[valid, 2] = np.nan

    return xyz


def create_pointcloud(rgb: np.ndarray, depth: np.ndarray, k_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    xyz = create_matrix_coordinate(depth, k_matrix)
    valid = np.isfinite(xyz[:, :, 0])

    # TODO robot inclination into consideration
    points = xyz[valid]
    colors = rgb[valid]
    return points, colors
