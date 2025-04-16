import logging
from typing import List, Tuple, Optional, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scripts.matching.topology_nav_graph import Node, Graph

logger = logging.getLogger(__name__)


def generate_match_grid(n: Node,
                        ugv_c: np.ndarray,
                        extracted_patches: List[np.ndarray],
                        correspondences_each_pairs: List[List[Tuple[np.ndarray, np.ndarray]]],
                        match_count_thresh: int) -> np.ndarray:
    grid_images = []
    for ugv_image, correspondences in zip(extracted_patches, correspondences_each_pairs):
        row_images = []
        for (img_path, uav_c, uav_feature), correspondence in zip(n.correspondance_data, correspondences):
            mkpts_0, mkpts_1 = correspondence
            uav_image = cv2.imread(img_path)

            ugv_corners_in_uav = None
            uav_corners_in_ugv = None
            if mkpts_0.shape[0] > match_count_thresh:
                ugv_h, ugv_w = ugv_image.shape[:2]
                uav_h, uav_w = uav_image.shape[:2]
                M, mask = cv2.findHomography(mkpts_0, mkpts_1,
                                             cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
                if M is not None:
                    ugv_corners_in_uav = cv2.perspectiveTransform(
                        np.array([[0, 0], [ugv_w - 1, 0], [ugv_w - 1, ugv_h - 1], [0, ugv_h - 1]],
                                 dtype=np.float32).reshape(-1, 1, 2), M)
                    uav_corners_in_ugv = cv2.perspectiveTransform(
                        np.array([[0, 0], [uav_w - 1, 0], [uav_w - 1, uav_h - 1], [0, uav_h - 1]],
                                 dtype=np.float32).reshape(-1, 1, 2), np.linalg.inv(M))

            if ugv_corners_in_uav is None:
                kp1 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in mkpts_0]
                kp2 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in mkpts_1]
                matches = [cv2.DMatch(i, i, 0) for i in range(len(mkpts_0))]
                match_img = cv2.drawMatches(ugv_image, kp1, uav_image, kp2,
                                            matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            else:
                uav_image = cv2.drawContours(uav_image, [ugv_corners_in_uav[:, 0, :].astype(np.int32)],
                                             -1, (255, 0, 0), 5)
                ugv_image_cnt = cv2.drawContours(ugv_image.copy(), [uav_corners_in_ugv[:, 0, :].astype(np.int32)],
                                                 -1, (255, 0, 0), 5)
                match_img = cv2.drawMatches(ugv_image_cnt, [], uav_image, [],
                                            [], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            text = f'{mkpts_0.shape[0]} matches. {uav_c[0]:.1f}, {uav_c[1]:.1f}. {np.linalg.norm(uav_c - ugv_c):.1f}'
            cv2.putText(match_img, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            row_images.append(match_img)
        grid_images.append(np.hstack(row_images))
    return np.vstack(grid_images)


def generate_patch_location(n: Node,
                            correspondences_each_pairs: List[List[Tuple[np.ndarray, np.ndarray]]],
                            match_count_thresh: int) -> Optional[np.ndarray]:
    original_images: Dict[str, np.ndarray] = {}

    for correspondences in correspondences_each_pairs:
        for (img_path, uav_c, uav_feature), correspondence in zip(n.correspondance_data, correspondences):
            _, mkpts_1 = correspondence
            if mkpts_1.shape[0] > match_count_thresh:
                _, _, original_path, patch_center, patch_width, patch_angle = \
                    list(filter(lambda p: p[1] == img_path, n.patches))[0]

                if original_path not in original_images:
                    original_images[original_path] = cv2.imread(original_path)
                original_image = original_images[original_path]

                cv2.drawContours(original_image,
                                 [cv2.boxPoints((patch_center, (patch_width, patch_width), patch_angle)).astype(int)],
                                 0,
                                 (0, 255, 0), 2)

    return np.hstack(list(original_images.values())) if len(original_images.values()) else None


class RobotAnimator:
    def __init__(self, graph: Graph, match_count_thresh: int, match_count_probable_thresh: int, figsize=(8, 8)):
        self.graph = graph
        self.match_count_thresh = match_count_thresh
        self.match_count_probable_thresh = match_count_probable_thresh

        # Matplotlib figure setup
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.canvas = FigureCanvas(self.fig)
        self.ax.set_aspect('equal')

        # Data holders
        self.path_x = []
        self.path_y = []
        self.yaws = []
        self.node: Optional[Node] = None
        self.correspondence_count: List[bool] = []

    def update_robot_pose(self, x, y, yaw):
        self.path_x.append(x)
        self.path_y.append(y)
        self.yaws.append(yaw)

    def update_node_display(self, n: Node, count: List[int]):
        self.node = n
        self.correspondence_count = count

    def render(self):
        self.ax.cla()

        # draw nodes and patches
        for n in self.graph.nodes:
            cx, cy = n.coordinate
            r = n.radius
            circ = plt.Circle((cx, cy), r, color='gray', fill=False)
            self.ax.add_patch(circ)
            if n == self.node:
                for (_, pt, _), count in zip(n.correspondance_data, self.correspondence_count):
                    c = (1.,0.,0.)
                    if count > self.match_count_probable_thresh:
                        c = (1., 0.65, 0.)
                    if count > self.match_count_thresh:
                        c = (0., 1., 0.)
                    rect = plt.Rectangle((pt[0] - 1, pt[1] - 1), 2, 2,
                                         color=c)
                    self.ax.add_patch(rect)
            else:
                for _, pt, _ in n.correspondance_data:
                    rect = plt.Rectangle((pt[0] - 1, pt[1] - 1), 2, 2, color=(1., 0., 0.))
                    self.ax.add_patch(rect)

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
