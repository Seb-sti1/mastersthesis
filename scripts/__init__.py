"""
This file defined helper functions
"""
import os
from pathlib import Path
from typing import Iterator, Tuple, Callable, Optional, Dict

import cv2
import numpy as np
from matplotlib import pyplot as plt


def resize_image(img: np.ndarray, max_width: Optional[int] = None, max_height: Optional[int] = None) -> np.ndarray:
    if max_width is None:
        max_width = 100_000
    if max_height is None:
        max_height = 100_000

    ratio = min(max_width / img.shape[0], max_height / img.shape[1], 1.0)
    return cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)


def load_paths(folder_path: str, extension: Callable[[str], bool]) -> Iterator[Path]:
    for file_path in sorted(filter(extension, os.listdir(folder_path))):
        yield Path(os.path.join(folder_path, file_path))


def load_paths_and_files(folder_path: str, extension: Callable[[str], bool],
                         max_width: Optional[int] = None,
                         max_height: Optional[int] = None) -> Iterator[Tuple[Path, np.ndarray]]:
    if max_width is None and max_height is None:
        for file_path in load_paths(folder_path, extension):
            yield file_path, cv2.imread(str(file_path))
    else:
        for file_path in load_paths(folder_path, extension):
            yield file_path, resize_image(cv2.imread(str(file_path)), max_width, max_height)


def load_files(folder_path: str, extension: Callable[[str], bool],
               max_width: Optional[int] = None,
               max_height: Optional[int] = None) -> Iterator[np.ndarray]:
    for _, img in load_paths_and_files(folder_path, extension, max_width, max_height):
        yield img


def show_image(image: np.ndarray, title: str = "An image", clusters: Optional[Dict] = None, save=False) -> None:
    plt.imshow(image)
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


def show_images(images: list[np.ndarray], grid_size: Tuple[int, int], title: str = "An image", save=False) -> None:
    n, m = grid_size
    fig, axes = plt.subplots(n, m, figsize=(m * 3, n * 3))  # Adjust size as needed
    plt.title(title)
    axes = axes.flatten()  # Flatten the 2D axes array to make indexing easier
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
        ax.axis('off')  # Turn off axes for empty subplots
    plt.tight_layout()
    if save:
        plt.savefig(f"{title}.png")
    else:
        plt.show()


def get_dataset_by_name(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "datasets", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {name} not found at {path}")
    return path
