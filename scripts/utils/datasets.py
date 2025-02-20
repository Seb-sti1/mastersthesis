"""
This file defined helper functions to manipulate files in datasets.
"""
import logging
import os
from pathlib import Path
from typing import Iterator, Tuple, Callable, Optional, TypeVar, Dict, Union

import cv2
import numpy as np

T = TypeVar("T")
FileLoader = Callable[[str], T]

known_file_loaders: Dict[str, FileLoader] = {
    ".png": cv2.imread,
    ".jpg": cv2.imread,
    ".JPG": cv2.imread,
    ".jpeg": cv2.imread,
    ".npy": np.load
}


def get_dataset_by_name(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {name} not found at {path}")
    return path


def resize_image(img: np.ndarray, max_width: Optional[int] = None, max_height: Optional[int] = None) -> np.ndarray:
    if max_width is None:
        max_width = 100_000
    if max_height is None:
        max_height = 100_000

    ratio = min(max_width / img.shape[0], max_height / img.shape[1], 1.0)
    return cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)


def load_paths_and_files(folder_path: Union[str, Path],
                         extension_filter: Optional[Callable[[str], bool]] = None,
                         max_width: Optional[int] = None,
                         max_height: Optional[int] = None,
                         additional_file_loaders: Optional[Dict[str, FileLoader]] = None) -> Iterator[Tuple[Path, T]]:
    """
    Load files from folder_path, filtered by the func extension_filter.
    The files are loaded using the known_file_loaders, it is possible to add more loaders using the additional_file_loaders parameter.
    If max_width and max_height are not None, the images are resized to fit the constraints.
    :param folder_path: the folder of the files
    :param extension_filter: a function to filter the files
    :param max_width: the maximum width of the images
    :param max_height: the maximum height of the images
    :param additional_file_loaders: additional file loaders (association of extension and function)
    :return: an iterator of tuples (filepath, file)
    """
    files = sorted(
        os.listdir(folder_path) if extension_filter is None else filter(extension_filter, os.listdir(folder_path)))
    file_loaders = known_file_loaders.copy()
    if additional_file_loaders is not None:
        file_loaders.update(additional_file_loaders)

    for filepath in files:
        filepath = Path(os.path.join(folder_path, filepath))
        if filepath.suffix not in file_loaders:
            logging.warning("File extension %s not supported", filepath.suffix)
            continue
        file = file_loaders[filepath.suffix](str(filepath))
        yield filepath, file if max_width is None and max_height is None else resize_image(file, max_width, max_height)


def load_files(folder_path: Union[str, Path],
               extension_filter: Optional[Callable[[str], bool]] = None,
               max_width: Optional[int] = None,
               max_height: Optional[int] = None,
               additional_file_loaders: Optional[Dict[str, FileLoader]] = None) -> Iterator[T]:
    """
    Load files from folder_path, filtered by the func extension_filter.
    The files are loaded using the known_file_loaders, it is possible to add more loaders using the additional_file_loaders parameter.
    If max_width and max_height are not None, the images are resized to fit the constraints.
    :param folder_path: the folder of the files
    :param extension_filter: a function to filter the files
    :param max_width: the maximum width of the images
    :param max_height: the maximum height of the images
    :param additional_file_loaders: additional file loaders (association of extension and function)
    :return: an iterator of the files
    """
    return map(lambda x: x[1], load_paths_and_files(folder_path,
                                                    extension_filter,
                                                    max_width, max_height,
                                                    additional_file_loaders))
