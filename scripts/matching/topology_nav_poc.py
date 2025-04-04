from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from matching.generate_test_data import get_aruco, create_matrix_front_patch, apply_rotations, extract_patch
from scripts.utils.datasets import get_dataset_by_name, load_paths_and_files, resize_image
from scripts.utils.norlab_sync_viz import seq1

type Coordinate = np.ndarray


class Node:
    def __init__(self, name, coordinate: Coordinate, radius: float) -> None:
        self.name = name
        self.coordinate = coordinate
        self.radius = radius
        self.patches = []
        self.path = Path(get_dataset_by_name("norlab_ulaval_datasets/node_dataset")) / self.name
        self.path.mkdir(parents=True, exist_ok=True)

    def distance(self, c: Coordinate) -> float:
        return float(np.linalg.norm(self.coordinate - c))

    def is_in(self, c: Coordinate) -> bool:
        return self.distance(c) < self.radius

    def add_patch(self, image: np.ndarray, c: Coordinate, orientation: float) -> None:
        cv2.imwrite(str(self.path / f"{len(self.patches)}.png"), image)
        self.patches.append((c, orientation, str(self.path / f"{len(self.patches)}.png")))

    def save_metadata(self):
        df = pd.DataFrame([(c[0], c[1], o, path) for (c, o, path) in self.patches], columns=["x", "y", "yaw", "path"])
        df.to_csv(str(self.path / "metadata.csv"), index=False)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = {}

    def add_node(self, node: Node):
        """
        Add a node to the graph
        :param node: the node to add
        """
        self.nodes.append(node)
        self.edges[node] = []

    def add_edge(self, n1: Node, n2: Node):
        """
        Add an edge between two nodes. This is a non-oriented graph.
        :param n1: the first node
        :param n2: the second node
        """
        if n1 not in self.edges:
            self.add_node(n1)
        if n2 not in self.edges[n1]:
            self.edges[n1].append(n2)
        if n2 not in self.edges:
            self.add_node(n2)
        if n1 not in self.edges[n2]:
            self.edges[n2].append(n1)

    def plot(self):
        fig, ax = plt.subplots()
        ax.add_collection(matplotlib.collections.PatchCollection([plt.Circle(n.coordinate,
                                                                             radius=n.radius,
                                                                             linewidth=0) for n in self.nodes],
                                                                 facecolor='purple'))

        for n1 in self.edges:
            for n2 in self.edges[n1]:
                x_values = [n1.coordinate[0], n2.coordinate[0]]
                y_values = [n1.coordinate[1], n2.coordinate[1]]
                ax.plot(x_values, y_values, color='black')  # Ensure color is set for visibility

        return fig

    def get_current_node(self, c: Coordinate) -> Optional[Node]:
        for n in self.nodes:
            if n.is_in(c):
                return n

    def save(self):
        path = Path(get_dataset_by_name("norlab_ulaval_datasets/node_dataset")) / "graph.csv"
        df = pd.DataFrame([(n.name, n.coordinate[0], n.coordinate[1], n.radius,
                            ",".join([m.name for m in self.edges[n]])) for n in self.nodes],
                          columns=["name", "x", "y", "r", "connected"])
        df.to_csv(path, index=False)

    def load(self):
        path = Path(get_dataset_by_name("norlab_ulaval_datasets/node_dataset")) / "graph.csv"
        df = pd.read_csv(str(path))

        constructed_node = {}
        for _, row in df.iterrows():
            constructed_node[str(row["name"])] = Node(str(row["name"]), (row["x"], row["y"]), row["r"])
            self.add_node(constructed_node[str(row["name"])])

        for _, row in df.iterrows():
            for connected_node_name in row["connected"].split(","):
                self.add_edge(constructed_node[str(row["name"])], constructed_node[connected_node_name])


def create_norlab_graph():
    g = Graph()
    n1 = Node("1", np.array([3, 7]), 5)
    n2 = Node("2", np.array([10, 18]), 3)
    g.add_edge(n1, n2)
    n3 = Node("3", np.array([-1, 25.5]), 3)
    g.add_edge(n2, n3)
    n4 = Node("4", np.array([-25, 35]), 4)
    g.add_edge(n3, n4)
    n5 = Node("5", np.array([-32, 26]), 3)
    g.add_edge(n4, n5)
    n6 = Node("6", np.array([-40, 10]), 3)
    g.add_edge(n5, n6)
    n7 = Node("7", np.array([-8, -4]), 3)
    g.add_edge(n6, n7)
    g.add_edge(n7, n1)
    n8 = Node("8", np.array([2, -12]), 4)
    g.add_edge(n7, n8)
    n9 = Node("9", np.array([-10, 14]), 3)
    g.add_edge(n1, n9)
    g.add_edge(n9, n5)
    n10 = Node("10", np.array([7, 32]), 3)
    g.add_edge(n2, n10)
    g.add_edge(n3, n10)
    n11 = Node("11", np.array([19, 11]), 3)
    g.add_edge(n2, n11)
    n12 = Node("12", np.array([21, 35]), 3)
    g.add_edge(n11, n12)
    n13 = Node("13", np.array([-16, 47]), 3)
    g.add_edge(n12, n13)
    g.add_edge(n13, n4)
    return g


def overlap(rect1: np.ndarray, rect2: np.ndarray) -> bool:
    for p in rect1:
        if cv2.pointPolygonTest(rect2.reshape(-1, 1, 2), p.astype(np.float32), False) >= 0:
            return True
    for q in rect2:
        if cv2.pointPolygonTest(rect1.reshape(-1, 1, 2), q.astype(np.float32), False) >= 0:
            return True
    return False


def generate_scouting_data(vis):
    gnss = pd.read_csv(get_dataset_by_name("norlab_ulaval_datasets/test_dataset/sequence1/rtk_odom/rtk_odom.csv"))

    g = create_norlab_graph()
    g.save()
    # g.plot()
    # plot(gnss['x'], gnss['y'])

    path_in_nodes = []
    for x, y in zip(gnss['x'], gnss['y']):
        n = g.get_current_node(np.array([x, y]))
        if n is not None and (len(path_in_nodes) == 0 or path_in_nodes[-1] != n):
            path_in_nodes.append(n)
    print(path_in_nodes)

    is_running = False

    for filepath, image in tqdm(load_paths_and_files(get_dataset_by_name(seq1 / "aerial" / "images"))):
        uav_time = int(filepath.stem)
        search_gnss = gnss.iloc[(gnss['timestamp'] - uav_time).abs().argmin()]
        uav_2d_position = np.array([search_gnss['x'], search_gnss['y']])

        aruco = get_aruco(image)
        if aruco is None:
            """
            The aruco is required when using the norlab datasets as there is no estimate of the heading 
            of the uav. This is compensated by using the orientation of the aruco tag.
            In the final algorithm, this trick won't be necessary.
            """
            continue
        perimeter = np.sum(np.array([np.linalg.norm(aruco[0, i, :] - aruco[0, i + 1, :]) for i in range(-1, 3)]))
        pixels_per_meter = perimeter / 1.4

        aruco_angle = np.arctan2(aruco[0, 1, 1] - aruco[0, 0, 1],
                                 aruco[0, 1, 0] - aruco[0, 0, 0])  # angle from ugv to uav
        yaw_ugv = search_gnss['yaw']  # angle from global to ugv
        yaw_uav = yaw_ugv + aruco_angle  # angle from global to uav
        yaw = - yaw_uav  # angle from uav to global
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])

        patches_width = 480  # 480 ~= pixels_per_meter*2
        patches = create_matrix_front_patch(patches_width,
                                            image.shape[:2],
                                            pattern=(4, 7),
                                            margin=(20, 50))
        c = np.mean(aruco[0, :, :], axis=0)
        w, h = 250, 325
        exclusion_rect = np.array([c + [-w, -h], c + [w, -h], c + [w, h], c + [-w, h]]) + [0, -100]
        exclusion_rect = apply_rotations([exclusion_rect], [-3])[0].astype(np.int32)

        visible_nodes = []
        for p in patches:
            center = (np.array(image.shape)[:2] // 2 - np.mean(p, axis=0)[::-1]) / pixels_per_meter
            coordinate = uav_2d_position + R @ center
            n = g.get_current_node(coordinate)
            if n is not None and not overlap(p, exclusion_rect):
                visible_nodes.append(n)
                n.add_patch(extract_patch(p, image, patches_width), coordinate, yaw_uav)
            else:
                visible_nodes.append(None)

        if vis:
            cv2.polylines(image, [exclusion_rect], isClosed=True, color=(0, 0, 255), thickness=10)
            for p, n in zip(patches, visible_nodes):
                c = (0, 0, 255) if n is None else (0, 255, 0)
                cv2.polylines(image, [p], isClosed=True, color=c, thickness=10)
                cv2.putText(image,
                            f"{str(n)}",
                            np.mean(p, axis=0).astype(np.int32),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 7)
            cv2.imshow("image", resize_image(image, 600))
            k = cv2.waitKey(5 if is_running else 0) & 0xFF
            if k == ord('q'):
                break
            elif k == ord(' '):
                is_running = not is_running

    for node in g.nodes:
        print(f"{node}: {len(node.patches)}")
        node.save_metadata()


# TODO iterate over images
#   - get patch up, down, right left (square) (/!\ ignoring uav)
#   - get true coordinate of each patch using aruco, exclude patch not in node
#   - get true orientation using aruco
#   - extract score using xfeat: threshold using value
#   - save image patch (debug) and feature (for uav nav)


if __name__ == "__main__":
    generate_scouting_data(False)
