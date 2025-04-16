from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from scripts.utils.datasets import get_dataset_by_name

default_path = Path(get_dataset_by_name("norlab_ulaval_datasets/node_dataset"))
logger = logging.getLogger(__name__)


class Node:
    def __init__(self, name, coordinate: np.ndarray, radius: float) -> None:
        # Node core definition
        self.name = name
        self.coordinate = coordinate
        self.radius = radius
        # Node data
        self.patches: List[Tuple[np.ndarray, str, str, np.ndarray, float, float]] = []
        self.correspondance_data: List[Tuple[str, np.ndarray, str]] = []
        # Node path
        self.path = default_path / self.name
        self.patches_metadata_path = self.path / "patches_metadata.csv"
        self.feat_metadata_path = self.path / "feat_metadata.csv"
        self.path.mkdir(parents=True, exist_ok=True)

    def distance(self, c: np.ndarray) -> float:
        return float(np.linalg.norm(self.coordinate - c))

    def is_in(self, c: np.ndarray) -> bool:
        return self.distance(c) < self.radius

    def add_patch(self, image: np.ndarray, c: np.ndarray,
                  original_path: str, patch_center: np.ndarray, patch_width: float, patch_angle: float) -> None:
        cv2.imwrite(str(self.path / f"{len(self.patches)}.png"), image)
        self.patches.append((c, str(self.path / f"{len(self.patches)}.png"),
                             original_path, patch_center, patch_width, patch_angle))

    def save(self):
        if len(self.patches) > 0:
            df = pd.DataFrame([(c[0], c[1], path,
                                original_path, patch_center[0], patch_center[1], patch_width, patch_angle)
                               for (c, path, original_path, patch_center, patch_width, patch_angle) in self.patches],
                              columns=["x", "y", "path",
                                       "original_path", "x_c", "y_c", "width", "angle"])
            df.to_csv(str(self.patches_metadata_path), index=False)

        if len(self.correspondance_data) > 0:
            for i, (_, _, feat) in enumerate(self.correspondance_data):
                with open(str(self.path / f"feat_{i}.pkl"), "wb") as f:
                    pickle.dump(feat, f)
            df = pd.DataFrame([(image_path, c[0], c[1], str(self.path / f"feat_{i}.pkl"))
                               for i, (image_path, c, _) in enumerate(self.correspondance_data)],
                              columns=["image_path", "x", "y", "feat_path"])
            df.to_csv(str(self.feat_metadata_path), index=False)

    def load(self):
        if self.patches_metadata_path.exists():
            df = pd.read_csv(str(self.patches_metadata_path))
            self.patches = [(np.array([r['x'], r['y']]), r['path'],r['original_path'],
                             np.array([r['x_c'], r['y_c']]), r["width"], r["angle"]) for _, r in df.iterrows()]

        if self.feat_metadata_path.exists():
            df = pd.read_csv(str(self.feat_metadata_path))
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
        self.nodes: List[Node] = []
        self.edges: Dict[Node, List[Node]] = {}

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

    def plot(self, ax: Axes) -> Axes:
        ax.add_collection(matplotlib.collections.PatchCollection(
            [plt.Circle(n.coordinate, radius=n.radius, linewidth=0) for n in self.nodes],
            facecolor='purple'
        ))

        for n1 in self.edges:
            for n2 in self.edges[n1]:
                x_values = [n1.coordinate[0], n2.coordinate[0]]
                y_values = [n1.coordinate[1], n2.coordinate[1]]
                ax.plot(x_values, y_values, color='black')

        return ax

    def get_current_node(self, c: np.ndarray) -> Optional[Node]:
        for n in self.nodes:
            if n.is_in(c):
                return n
        return None

    def save(self):
        path = default_path / "graph.csv"
        df = pd.DataFrame([(n.name, n.coordinate[0], n.coordinate[1], n.radius,
                            ",".join([m.name for m in self.edges[n]])) for n in self.nodes],
                          columns=["name", "x", "y", "r", "connected"])
        df.to_csv(path, index=False)

        for n in self.nodes:
            n.save()

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

        for n in self.nodes:
            n.load()
