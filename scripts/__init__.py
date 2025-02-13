"""
This file defined helper functions
"""
import os
from pathlib import Path
from typing import Iterator, Tuple, Callable, Optional, Dict

import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_path_and_files(folder_path: str, extension: Callable[[str], bool]) -> Iterator[Tuple[Path, np.ndarray]]:
    for file_path in sorted(filter(extension, os.listdir(folder_path))):
        yield Path(os.path.join(folder_path, file_path)), cv2.imread(os.path.join(folder_path, file_path))


def load_files(folder_path: str, extension: Callable[[str], bool]) -> Iterator[np.ndarray]:
    for file_path in sorted(filter(extension, os.listdir(folder_path))):
        yield cv2.imread(os.path.join(folder_path, file_path))


def resize_images(images: Iterator[np.ndarray],
                  max_width: Optional[int] = None, max_height: Optional[int] = None) -> Iterator[np.ndarray]:
    if max_width is None:
        max_width = 100_000
    if max_height is None:
        max_height = 100_000

    for img in images:
        ratio = min(max_width / img.shape[0], max_height / img.shape[1], 1.0)
        yield cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)


def show_image(image: np.ndarray, title: str = "An image", clusters: Optional[Dict] = None) -> None:
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
    plt.show()


def show_images(images: list[np.ndarray], grid_size: Tuple[int, int], title: str = "An image") -> None:
    n, m = grid_size
    fig, axes = plt.subplots(n, m, figsize=(m * 3, n * 3))  # Adjust size as needed
    plt.title(title)
    axes = axes.flatten()  # Flatten the 2D axes array to make indexing easier
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
        ax.axis('off')  # Turn off axes for empty subplots
    plt.tight_layout()
    plt.show()
