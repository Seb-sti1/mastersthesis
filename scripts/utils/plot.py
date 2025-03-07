"""
This module contains utility functions for plotting data.
"""
import importlib
import threading
import time
from typing import Optional, Dict, Tuple, List

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


def get_color_map(n: int) -> np.ndarray:
    cmap_desc = plt.get_cmap("hsv", n)
    return (cmap_desc(np.arange(n))[:, :3] * 255).astype(np.uint8)


def plot_histogram(data: np.ndarray, bins: int = 30, title: str = "Histogram of Values") -> None:
    values = data.flatten()
    values = values[np.isfinite(values)]
    plt.hist(values, bins=bins, edgecolor='black')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def plot(x: np.ndarray | list, y: np.ndarray | list,
         xlabel: str = "x", ylabel: str = "y",
         title: str = "A plot") -> None:
    plt.title(title)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_bar(groups_name: list[str],
             features: Dict[str, np.ndarray | list],
             xlabel: str = "x", ylabel: str = "y",
             title: str = "A plot") -> None:
    x = np.arange(len(groups_name))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in features.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width, groups_name)
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
        while not self.should_close.is_set():
            if self.should_update.is_set():
                self.should_update = threading.Event()
                # update cloud
                self.vis.clear_geometries()
                self.vis.add_geometry(self.cloud, reset_bounding_box=self.first)
                self.vis.add_geometry(axis)
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
