"""
This is a test script for the Detectree model, see https://github.com/martibosch/detectree?tab=readme-ov-file
https://huggingface.co/martibosch/detectree
"""
import argparse
import hashlib
import logging
import os.path
import tempfile
import time

import cv2
import detectree as dtr
import numpy as np
from tqdm import tqdm

from scripts import get_dataset_by_name, load_paths_and_files, show_images, RamMonitor

DATASET_PATH = get_dataset_by_name("aukerman")
BENCHMARK_EXPORT_PATH = os.path.join(os.path.dirname(__file__), f"benchmark_results_{time.time()}.csv")
logger = logging.getLogger(__name__)


def hash_int(s: str) -> int:
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8


def make_predictions(img: np.array) -> np.array:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        cv2.imwrite(temp_file.name, img)
        return dtr.Classifier().predict_img(temp_file.name)


def classify(max_image_size: int) -> None:
    for path, img in load_paths_and_files(DATASET_PATH, lambda x: x.endswith(".JPG"),
                                          max_width=max_image_size,
                                          max_height=max_image_size):
        y_pred = make_predictions(img)
        show_images([y_pred, img], (1, 2), title=f"Image: {path.name}")
        input("Press Enter to continue...")


def benchmark(max_image_size: int, count_img: int) -> list[list[float | int]]:
    r = []
    for path, img in load_paths_and_files(DATASET_PATH, lambda x: x.endswith(".JPG"),
                                          max_width=max_image_size,
                                          max_height=max_image_size):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            cv2.imwrite(temp_file.name, img)
            ram = RamMonitor()
            ram.start()
            start = time.time()
            y_pred = dtr.Classifier().predict_img(temp_file.name)
            end = time.time()
            ram.stop()
            r.append([
                end - start,
                hash_int(path.name),
                img.shape[0] * img.shape[1],
                max_image_size,
                ram.get_max_ram(),
                ram.get_average_ram()
            ])
            show_images([y_pred, img], (1, 2), title=f"{max_image_size}_{path.name}", save=True)
        if len(r) == count_img:
            break

    return r


def main(args=None) -> None:
    parser = argparse.ArgumentParser(args)
    parser.add_argument("--benchmark", "-b", action='store_true', default=False,
                        help="Run benchmark test")
    parser.add_argument("--max-image-size",
                        type=int, default=256,
                        help="Maximum size of the image's width or height")
    args = parser.parse_args()

    if args.benchmark:
        results = []
        for max_image_size in tqdm([100, 200, 400, 800, 1000]):
            results += benchmark(max_image_size, 3)
        np.savetxt(BENCHMARK_EXPORT_PATH, np.array(results), delimiter="\t")
    else:
        classify(args.max_image_size)


if __name__ == "__main__":
    main()
