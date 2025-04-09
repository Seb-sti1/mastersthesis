from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Callable, Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy import floating
from tqdm import tqdm

from matching.generate_test_data import get_aruco, create_matrix_front_patch, apply_rotations
from scripts.utils.datasets import get_dataset_by_name, load_paths_and_files, resize_image
from scripts.utils.norlab_sync_viz import seq1

_initiated_xfeat: Optional[torch] = None
default_path = Path(get_dataset_by_name("norlab_ulaval_datasets/node_dataset"))
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
type Coordinate = np.ndarray


def init_xfeat():
    global _initiated_xfeat
    logger.debug(f"Cuda device(s) {os.environ['CUDA_VISIBLE_DEVICES']}"
                 if "CUDA_VISIBLE_DEVICES" in os.environ else "No CUDA devices")
    _initiated_xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)


def get_xfeat():
    global _initiated_xfeat
    if _initiated_xfeat is None:
        init_xfeat()
    return _initiated_xfeat


class Node:
    def __init__(self, name, coordinate: Coordinate, radius: float) -> None:
        self.name = name
        self.coordinate = coordinate
        self.radius = radius
        self.patches = []
        self.path = default_path / self.name
        self.path.mkdir(parents=True, exist_ok=True)
        self.correspondance_data = []

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

    def load_metadata(self):
        df = pd.read_csv(str(self.path / "metadata.csv"))
        self.patches = [((r['x'], r['y']), r['yaw'], r['path']) for _, r in df.iterrows()]

    def add_correspondance(self,
                           features: np.ndarray,
                           patch: Tuple[Coordinate, float, str],
                           memory_size: int,
                           scoring_function: Callable[[Coordinate, float, np.ndarray], floating[Any]]) -> None:
        self.correspondance_data.append((patch[0], patch[1], features))
        if len(self.correspondance_data) > memory_size:
            # FIXME test this
            self.correspondance_data = sorted(self.correspondance_data, key=lambda x: scoring_function(*x),
                                              reverse=True)
            self.correspondance_data = self.correspondance_data[:memory_size]

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
        path = default_path / "graph.csv"
        df = pd.DataFrame([(n.name, n.coordinate[0], n.coordinate[1], n.radius,
                            ",".join([m.name for m in self.edges[n]])) for n in self.nodes],
                          columns=["name", "x", "y", "r", "connected"])
        df.to_csv(path, index=False)

    def load(self):
        path = default_path / "graph.csv"
        df = pd.read_csv(str(path))

        constructed_node = {}
        for _, row in df.iterrows():
            constructed_node[str(row["name"])] = Node(str(row["name"]), (row["x"], row["y"]), row["r"])
            self.add_node(constructed_node[str(row["name"])])
            constructed_node[str(row["name"])].load_metadata()

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


def generate_scouting_data(g: Graph, gnss: pd.DataFrame, vis: bool):
    is_running = False
    for filepath, image in tqdm(load_paths_and_files(get_dataset_by_name(seq1 / "aerial" / "images"))):
        aruco = get_aruco(image)
        if aruco is None:
            """
            The aruco is required when using the norlab datasets as there is no estimate of the heading 
            of the uav. This is compensated by using the orientation of the aruco tag.
            In the final algorithm, this trick won't be necessary.
            """
            continue
        aruco = aruco[0, :, :]  # there is only one aruco tag
        aruco_a = np.arctan2(aruco[0, 1] - aruco[1, 1],
                             aruco[1, 0] - aruco[0, 0]) - np.pi / 2  # angle from ugv to uav
        aruco_c = np.mean(aruco, axis=0)  # position of the aruco in the image
        pixels_per_meter = np.sum(np.array([np.linalg.norm(aruco[i, :] - aruco[i + 1, :])
                                            for i in range(-1, 3)])) / 1.4  # resolution of the image

        uav_time = int(filepath.stem)
        search_gnss = gnss.iloc[(gnss['timestamp'] - uav_time).abs().argmin()]
        tf_ugv_to_map = np.linalg.inv(get_transformation_matrix(np.array([search_gnss['x'], search_gnss['y']]),
                                                                search_gnss['yaw']))

        patches_width = 480  # 480 ~= pixels_per_meter*2
        patches = create_matrix_front_patch(patches_width,
                                            image.shape[:2],
                                            pattern=(4, 7),
                                            margin=(20, 50))
        # patches = apply_rotations(patches, [-yaw_uav])
        w, h = 250, 325
        exclusion_rect = apply_rotations([np.array([aruco_c + [-w, -h], aruco_c + [w, -h],
                                                    aruco_c + [w, h], aruco_c + [-w, h]]) + [0, -100]],
                                         [-3])[0].astype(np.int32)

        visible_nodes = []
        # for p in patches:
        #     from_aruco_to_p_in_tf_uav = (aruco_origin - np.mean(p, axis=0)[::-1]) / pixels_per_meter
        #     from_aruco_to_p_in_tf_ugv = rotation_matrix(-aruco) @ from_aruco_to_p_in_tf_uav
        #     coordinate = uav_2d_position + from_aruco_to_p_in_tf_uav
        #     n = g.get_current_node(coordinate)
        #     if n is not None and not overlap(p, exclusion_rect):
        #         visible_nodes.append(n)
        #         n.add_patch(extract_patch(p, image, patches_width), coordinate, yaw_uav)
        #     else:
        #         visible_nodes.append(None)

        if vis:
            # draw aruco
            for i, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]):
                cv2.circle(image, aruco[i, :].astype(np.int32), radius=5, color=c, thickness=-1)

            # draw ugv frame
            for o, xy_list in zip([np.array([0, 0]),
                                   tf_ugv_to_map[:2, 2]],
                                  [[np.array([1, 0]), np.array([0, 1])],
                                   [(tf_ugv_to_map @ np.array([1, 0, 1]))[:2],
                                    (tf_ugv_to_map @ np.array([0, 1, 1]))[:2]]]):  # FIXME
                ij_o = from_ugv_to_pixel(o, aruco_a, aruco_c, pixels_per_meter)
                for xy, c in zip(xy_list, [(0, 0, 255), (0, 255, 0)]):
                    ij = from_ugv_to_pixel(xy, aruco_a, aruco_c, pixels_per_meter)
                    cv2.line(image, ij_o, ij, c, 10)

            # draw exclusion zone and patches
            cv2.polylines(image, [exclusion_rect], isClosed=True, color=(0, 0, 255), thickness=10)
            for p, n in zip(patches, visible_nodes):
                aruco_c = (0, 0, 255) if n is None else (0, 255, 0)
                cv2.polylines(image, [p], isClosed=True, color=aruco_c, thickness=10)
                cv2.putText(image,
                            f"{str(n)}",
                            np.mean(p, axis=0).astype(np.int32),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 7)

            # show image
            cv2.imshow("image", resize_image(image, 600))
            k = cv2.waitKey(5 if is_running else 0) & 0xFF
            if k == ord('q'):
                break
            elif k == ord(' '):
                is_running = not is_running

    for node in g.nodes:
        print(f"{node}: {len(node.patches)}")
        node.save_metadata()

    cv2.destroyAllWindows()


def filter_scouting_data(g: Graph, vis: bool):
    xfeat = get_xfeat()

    def scoring(_: Coordinate, __: float, feats: np.ndarray) -> floating[Any]:
        return np.median(feats['scores'].cpu().numpy())

    keep = 50
    for n in g.nodes:
        current_min_score = 0
        for p in tqdm(n.patches):
            image = cv2.imread(p[2])
            output = xfeat.detectAndCompute(image, top_k=4096)[0]
            output.update({'image_size': (image.shape[1], image.shape[0])})
            score = scoring(p[0], p[1], output)
            if score > current_min_score:
                current_min_score = min(score, current_min_score)
                n.add_correspondance(output, p, keep, scoring)

    return g


def detect_ugv_location(g: Graph, gnss: pd.DataFrame, vis: bool):
    is_running = False
    for (_, ugv_image), (filepath, ugv_bev) in tqdm(
            zip(load_paths_and_files(get_dataset_by_name(seq1 / "ground" / "images")),
                load_paths_and_files(get_dataset_by_name(seq1 / "ground" / "projections")))):
        ugv_time = int(filepath.stem)
        search_gnss = gnss.iloc[(gnss['timestamp'] - ugv_time).abs().argmin()]
        ugv_2d_position = np.array([search_gnss['x'], search_gnss['y']])
        yaw_ugv = search_gnss['yaw']  # angle from global to ugv

    pass


if __name__ == "__main__":

    if (default_path / "graph.csv").exists():
        graph = Graph()
        graph.load()
    else:
        graph = create_norlab_graph()
        graph.save()

    gnss_norlab = pd.read_csv(
        get_dataset_by_name("norlab_ulaval_datasets/test_dataset/sequence1/rtk_odom/rtk_odom.csv"))
    # graph.plot()
    # plot(gnss['x'], gnss['y'])

    path_in_nodes = []
    for x, y in zip(gnss_norlab['x'], gnss_norlab['y']):
        n = graph.get_current_node(np.array([x, y]))
        if n is not None and (len(path_in_nodes) == 0 or path_in_nodes[-1] != n):
            path_in_nodes.append(n)
    print(path_in_nodes)

    generate_scouting_data(graph, gnss_norlab, True)
    # graph = filter_scouting_data(graph, True)
    # detect_ugv_location(graph, gnss_norlab, True)
