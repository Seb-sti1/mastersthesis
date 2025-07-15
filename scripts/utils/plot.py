"""
This module contains utility functions for plotting data.
"""
import importlib
import threading
import time
from typing import Optional, Dict, Tuple, List, Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def show_image(image: np.ndarray, title: str = "An image", clusters: Optional[Dict] = None, save=False) -> None:
    plt.imshow(image, interpolation="nearest")
    plt.title(title)
    plt.axis('off')
    if clusters:
        handles = []
        for cluster in clusters.values():
            color = np.array(cluster['color']) / 255  # Normalize to [0,1] for Matplotlib
            patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10,
                               label=cluster['name'])
            handles.append(patch)
        plt.legend(handles=handles, loc="upper right")
    if save:
        plt.savefig(f"{title}.png")
    else:
        plt.show()


def show_images(images: List[np.ndarray], grid_size: Tuple[int, int], title: str = "An image", save=False) -> None:
    n, m = grid_size
    fig, axes = plt.subplots(n, m, figsize=(m * 3, n * 3))  # Adjust size as needed
    plt.title(title)
    axes = axes.flatten()  # Flatten the 2D axes array to make indexing easier
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], interpolation="nearest")
        ax.axis('off')  # Turn off axes for empty subplots
    plt.tight_layout()
    if save:
        plt.savefig(f"{title}.png")
    else:
        plt.show()


def get_color_map(n: int, cmap="hsv") -> np.ndarray:
    cmap_desc = plt.get_cmap(cmap, n)
    return (cmap_desc(np.arange(n))[:, :3] * 255).astype(np.uint8)


def plot_histogram(data: np.ndarray, bins: int = 30, title: str = "Histogram of Values") -> None:
    values = data.flatten()
    values = values[np.isfinite(values)]
    plt.hist(values, bins=bins, edgecolor='black')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def plot(x: Union[np.ndarray, List], y: Union[np.ndarray, List],
         xlabel: str = "x", ylabel: str = "y",
         title: str = "A plot",
         should_show=True) -> None:
    plt.title(title)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.grid()
    plt.axis("equal")
    if should_show:
        plt.show()


def plot_scatter_2axis(x: Union[np.ndarray, List], y1: Union[np.ndarray, List],
                       y2: Union[np.ndarray, List],
                       xlabel: str = "x", ylabel1: str = "y1", ylabel2: str = "y2",
                       ylim1: Optional[Tuple[float, float]] = None,
                       ylim2: Optional[Tuple[float, float]] = None,
                       title: str = "A plot") -> None:
    matplotlib.rcParams.update({'font.size': 18})
    fig, ax1 = plt.subplots()
    ax1.set_xlim((0, np.max(x) * 1.1))
    ax1.scatter(x, y1, color='b', label=ylabel1, alpha=0.7)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color='b')
    if ylim1 is not None:
        ax1.set_ylim(ylim1)
    ax1.tick_params(axis='y', labelcolor='b')
    plt.grid()

    ax2 = ax1.twinx()
    ax2.scatter(x, y2, color='r', label=ylabel2, alpha=0.7)
    ax2.set_ylabel(ylabel2, color='r')
    if ylim1 is not None:
        ax2.set_ylim(ylim2)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.ticklabel_format(axis='y', scilimits=[-3, 3])
    # plt.grid()
    # plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_bar(groups_name: List[str],
             features: Dict[str, Union[np.ndarray, List]],
             xlabel: str = "x", ylabel: str = "y",
             title: str = "A plot") -> None:
    x = np.arange(len(groups_name))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(layout='constrained')

    for multiplier, (attribute, measurement) in enumerate(features.items()):
        offset = width * (multiplier - len(features) // 2)
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x, groups_name)
    ax.legend(loc='upper left', ncols=3)
    plt.show()


class DynamicO3DWindow:
    def __init__(self):
        self.vis = None
        self.o3d = importlib.import_module("open3d")

        self.should_update = threading.Event()
        self.should_close = threading.Event()

        self.pcd_thread = threading.Thread(target=self.__show_pcd__)
        self.pcd_thread.start()

        self.first = True
        self.cloud = None

    def __show_pcd__(self):
        if self.vis is None:
            self.vis = self.o3d.visualization.Visualizer()
            self.vis.create_window(width=800, height=600)
        axis = self.o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        rot_axis = self.o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        T = np.eye(4)
        T[:3, :3] = self.o3d.geometry.get_rotation_matrix_from_quaternion([0.94, -0.01, -0.03, -0.33])
        rot_axis.transform(T)
        while not self.should_close.is_set():
            if self.should_update.is_set():
                self.should_update = threading.Event()
                # update cloud
                self.vis.clear_geometries()
                self.vis.add_geometry(self.cloud, reset_bounding_box=self.first)
                self.vis.add_geometry(axis)
                self.vis.add_geometry(rot_axis)
                self.first = False
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)

        self.vis.destroy_window()

    def show_pcd(self, cloud):
        self.cloud = cloud
        self.should_update.set()

    def finish(self):
        self.should_close.set()
        self.pcd_thread.join()
