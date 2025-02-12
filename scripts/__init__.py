"""
This file defined helper functions
"""
import os
from pathlib import Path
from typing import Iterator, Tuple, Callable, Optional

import cv2
import numpy as np


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
