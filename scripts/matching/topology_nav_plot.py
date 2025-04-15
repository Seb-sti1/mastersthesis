import logging
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scripts.matching.topology_nav_graph import Node, Graph

logger = logging.getLogger(__name__)


def generate_match_grid(n: Node,
                        ugv_c: np.ndarray,
                        extracted_patches: List[np.ndarray],
                        correspondances_each_pairs: List[List[Tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
    grid_images = []
    for extracted_patch, correspondances in zip(extracted_patches, correspondances_each_pairs):
        row_images = []
        for (img_path, uav_c, uav_feature), correspondance in zip(n.correspondance_data, correspondances):
            mkpts_0, mkpts_1 = correspondance
            full_img = cv2.imread(img_path)

            kp1 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in mkpts_0]
            kp2 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in mkpts_1]
            matches = [cv2.DMatch(i, i, 0) for i in range(len(mkpts_0))]
            match_img = cv2.drawMatches(extracted_patch, kp1, full_img, kp2,
                                        matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            text = f'{mkpts_0.shape[0]} matches. {uav_c[0]:.1f}, {uav_c[1]:.1f}. {np.linalg.norm(uav_c - ugv_c):.1f}'
            cv2.putText(match_img, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            row_images.append(match_img)
        grid_images.append(np.hstack(row_images))
    return np.vstack(grid_images)


class RobotAnimator:
    def __init__(self, graph: Graph, figsize=(8, 8)):
        self.graph = graph

        # Matplotlib figure setup
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.canvas = FigureCanvas(self.fig)
        self.ax.set_aspect('equal')

        # Data holders
        self.path_x = []
        self.path_y = []
        self.yaws = []
        self.node: Optional[Node] = None
        self.detected_correspondance: List[bool] = []

    def update_robot_pose(self, x, y, yaw):
        self.path_x.append(x)
        self.path_y.append(y)
        self.yaws.append(yaw)

    def update_node_display(self, n: Node, detected: List[bool]):
        self.node = n
        self.detected_correspondance = detected

    def render(self):
        self.ax.cla()

        # draw nodes and patches
        for n in self.graph.nodes:
            cx, cy = n.coordinate
            r = n.radius
            circ = plt.Circle((cx, cy), r, color='gray', fill=False)
            self.ax.add_patch(circ)
            if n == self.node:
                for (_, pt, _), detected in zip(n.correspondance_data, self.detected_correspondance):
                    self.ax.scatter(pt[0], pt[1], s=5, color=(0., 1., 0.) if detected else (1., 0., 0.))
            else:
                for _, pt, _ in n.correspondance_data:
                    self.ax.scatter(pt[0], pt[1], s=5, color=(1., 0., 0.))

        # draw edges
        for n1 in self.graph.edges:
            for n2 in self.graph.edges[n1]:
                x_values = [n1.coordinate[0], n2.coordinate[0]]
                y_values = [n1.coordinate[1], n2.coordinate[1]]
                self.ax.plot(x_values, y_values, color='black')

        # draw robot path
        self.ax.plot(self.path_x, self.path_y, 'b-')
        if self.path_x:
            x, y, yaw = self.path_x[-1], self.path_y[-1], self.yaws[-1]
            self.ax.arrow(x, y, 2 * np.cos(yaw), 2 * np.sin(yaw), head_width=0.5, color='g')

        # convert to image
        self.canvas.draw()
        img = np.frombuffer(self.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(self.canvas.get_width_height()[::-1] + (4,))
        return img[:, :, ::-1]
