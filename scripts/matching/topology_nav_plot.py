import logging
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scripts.matching.topology_nav_graph import Node

logger = logging.getLogger(__name__)


def generate_match_grid(n: Node,
                        ugv_c: np.ndarray,
                        extracted_patches: List[np.ndarray],
                        correspondances_each_pairs: List[List[Tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
    grid_images = []
    for (img_path, uav_c, uav_feature), correspondances in zip(n.correspondance_data, correspondances_each_pairs):
        row_images = []
        full_img = cv2.imread(img_path)
        for extracted_patch, correspondance in zip(extracted_patches, correspondances):
            mkpts_0, mkpts_1 = correspondance
            if mkpts_0.shape[0] == 0:
                h1, w1 = extracted_patch.shape[:2]
                h2, w2 = full_img.shape[:2]
                match_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            else:
                kp1 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in mkpts_0]
                kp2 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in mkpts_1]
                matches = [cv2.DMatch(i, i, 0) for i in range(len(mkpts_0))]

                match_img = cv2.drawMatches(
                    extracted_patch, kp1,
                    full_img, kp2,
                    matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
            text = f'{mkpts_0.shape[0]} matches. {uav_c[0]:.1f}, {uav_c[1]:.1f}. {np.linalg.norm(uav_c - ugv_c):.1f}'
            cv2.putText(match_img, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            row_images.append(match_img)
        grid_images.append(np.hstack(row_images))
    return np.vstack(grid_images)


class RobotAnimator:
    def __init__(self, bg_plot_func,
                 node_plot_origin=(0.05, 0.05), node_plot_size=(0.3, 0.3),
                 node_plot_scale=10, figsize=(8, 8)):
        # Matplotlib figure setup
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.canvas = FigureCanvas(self.fig)
        self.ax.set_aspect('equal')
        self.bg_plot_func = bg_plot_func

        # Inset (node) axes
        self.node_ax = self.fig.add_axes([*node_plot_origin, *node_plot_size])
        self.node_ax.set_aspect('equal')
        self.node_scale = node_plot_scale

        # Data holders
        self.path_x = []
        self.path_y = []
        self.yaws = []
        self.node: Optional[Node] = None
        self.detected_correspondance: List[bool] = []

        # Initial static background
        self.bg_plot_func(self.ax)

    def update_robot_pose(self, x, y, yaw):
        self.path_x.append(x)
        self.path_y.append(y)
        self.yaws.append(yaw)

    def update_node_display(self, n: Node, detected: List[bool]):
        self.node = n
        self.detected_correspondance = detected

    def render(self):
        self.ax.cla()
        self.bg_plot_func(self.ax)

        # Draw robot path
        self.ax.plot(self.path_x, self.path_y, 'b-')
        if self.path_x:
            x, y, yaw = self.path_x[-1], self.path_y[-1], self.yaws[-1]
            self.ax.arrow(x, y, 2 * np.cos(yaw), 2 * np.sin(yaw), head_width=0.5, color='g')

        # Node inset
        self.node_ax.cla()
        if self.node:
            cx, cy = self.node.coordinate
            r = self.node.radius
            self.node_ax.set_xlim(cx - self.node_scale, cx + self.node_scale)
            self.node_ax.set_ylim(cy - self.node_scale, cy + self.node_scale)
            circ = plt.Circle((cx, cy), r, color='gray', fill=False)
            self.node_ax.add_patch(circ)
            for pt, color in zip(self.node.correspondance_data, self.detected_correspondance):
                self.node_ax.plot(pt[0], pt[1], 'o', color=color)

        # Convert to image
        self.canvas.draw()
        img = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.canvas.get_width_height()[::-1] + (3,))
        return img
