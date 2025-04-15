import logging
import os
from typing import Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd
import torch

from scripts.matching.topology_nav_graph import Graph, Node

_initiated_xfeat: Optional[torch] = None
logger = logging.getLogger(__name__)


def init_xfeat():
    global _initiated_xfeat
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    logger.debug(f"Cuda device(s) {os.environ['CUDA_VISIBLE_DEVICES']}"
                 if "CUDA_VISIBLE_DEVICES" in os.environ else "No CUDA devices")
    _initiated_xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)


def get_xfeat():
    global _initiated_xfeat
    if _initiated_xfeat is None:
        init_xfeat()
    return _initiated_xfeat


def overlap(rect1: np.ndarray, rect2: np.ndarray) -> bool:
    for p in rect1:
        if cv2.pointPolygonTest(rect2.reshape(-1, 1, 2), p.astype(np.float32), False) >= 0:
            return True
    for q in rect2:
        if cv2.pointPolygonTest(rect1.reshape(-1, 1, 2), q.astype(np.float32), False) >= 0:
            return True
    if cv2.pointPolygonTest(rect1, rect2.mean(axis=0).astype(np.float32), False) >= 0:
        return True
    return False


def rotation_matrix(a: float) -> np.ndarray:
    return np.array([[np.cos(a), -np.sin(a)],
                     [np.sin(a), np.cos(a)]])


def get_transformation_matrix(xy: np.ndarray, a: float) -> np.ndarray:
    t = np.zeros((3, 3))
    t[:2, :2] = rotation_matrix(a)
    t[:2, 2] = xy
    t[2, 2] = 1
    return t


def from_ugv_to_pixel(xy: np.ndarray, aruco_a: float, aruco_c: np.ndarray, pixels_per_meter: float) -> np.array:
    return (aruco_c - (rotation_matrix(aruco_a) @ xy)[::-1] * pixels_per_meter).astype(np.int32)


def from_pixel_to_ugv(ij: np.ndarray, aruco_a: float, aruco_c: np.ndarray, pixel_per_meter: float) -> np.array:
    return rotation_matrix(-aruco_a) @ ((aruco_c - ij.astype(np.float32)) / pixel_per_meter)[::-1]


def get_path_in_node(gnss_norlab: pd.DataFrame, graph: Graph) -> Tuple[List[Node], List[Node]]:
    path_in_nodes = []
    left_node = True
    current_nodes = []
    for x, y in zip(gnss_norlab['x'], gnss_norlab['y']):
        n = graph.get_current_node(np.array([x, y]))
        current_nodes.append(n)
        if n is None:
            left_node = True
        elif left_node:
            path_in_nodes.append(n)
            left_node = False
    logger.info(path_in_nodes)

    # same as current_nodes but none value are replace by the next node the robot will be visiting
    next_nodes = []
    next_node_idx = 0
    next_node_idx_used = False
    for i in range(len(current_nodes)):
        if current_nodes[i] is None:
            next_nodes.append(path_in_nodes[next_node_idx] if next_node_idx < len(path_in_nodes) else None)
            next_node_idx_used = True
        else:
            if next_node_idx_used:
                next_node_idx += 1
                next_node_idx_used = False
            next_nodes.append(current_nodes[i])

    return current_nodes, next_nodes
