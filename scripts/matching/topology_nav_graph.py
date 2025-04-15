from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.utils.datasets import get_dataset_by_name

default_path = Path(get_dataset_by_name("norlab_ulaval_datasets/node_dataset"))
logger = logging.getLogger(__name__)


class Node:
    def __init__(self, name, coordinate: np.ndarray, radius: float) -> None:
        self.name = name
        self.coordinate = coordinate
        self.radius = radius
        self.path = default_path / self.name
        self.path.mkdir(parents=True, exist_ok=True)
        self.patches = []
        self.correspondance_data = []

    def distance(self, c: np.ndarray) -> float:
        return float(np.linalg.norm(self.coordinate - c))

    def is_in(self, c: np.ndarray) -> bool:
        return self.distance(c) < self.radius

    def add_patch(self, image: np.ndarray, c: np.ndarray) -> None:
        cv2.imwrite(str(self.path / f"{len(self.patches)}.png"), image)
        self.patches.append((c, str(self.path / f"{len(self.patches)}.png")))

    def save_patches_metadata(self):
        df = pd.DataFrame([(c[0], c[1], o, path) for (c, o, path) in self.patches], columns=["x", "y", "yaw", "path"])
        df.to_csv(str(self.path / "metadata.csv"), index=False)

    def load_patches_metadata(self):
        df = pd.read_csv(str(self.path / "metadata.csv"))
        self.patches = [(np.array([r['x'], r['y']]), r['yaw'], r['path']) for _, r in df.iterrows()]

    def save_correspondances(self):
        for i, (_, _, feat) in enumerate(self.correspondance_data):
            with open(str(self.path / f"feat_{i}.pkl"), "wb") as f:
                pickle.dump(feat, f)
        df = pd.DataFrame([(image_path, c[0], c[1], str(self.path / f"feat_{i}.pkl"))
                           for i, (image_path, c, _) in enumerate(self.correspondance_data)],
                          columns=["image_path", "x", "y", "feat_path"])
        df.to_csv(str(self.path / "feat_metadata.csv"), index=False)

    def load_correspondances(self):
        df = pd.read_csv(str(self.path / "feat_metadata.csv"))

        for _, row in df.iterrows():
            with open(row["feat_path"], "rb") as f:
                self.correspondance_data.append((row['image_path'],
                                                 np.array([row['x'], row['y']]),
                                                 pickle.load(f)))

    def __str__(self):
        return f"Node({self.name})"

    def __repr__(self):
        return str(self)


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

    def get_current_node(self, c: np.ndarray) -> Optional[Node]:
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

        for _, row in df.iterrows():
            for connected_node_name in row["connected"].split(","):
                self.add_edge(constructed_node[str(row["name"])], constructed_node[connected_node_name])
