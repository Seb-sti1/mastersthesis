from typing import Optional, Dict, Tuple

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


def show_images(images: list[np.ndarray], grid_size: Tuple[int, int], title: str = "An image", save=False) -> None:
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
