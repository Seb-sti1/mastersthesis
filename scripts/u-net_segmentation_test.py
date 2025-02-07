"""
This uses https://github.com/milesial/Pytorch-UNet?tab=readme-ov-file#pretrained-model to try to segment the images
from BEV images of different datasets

Current results: The pretrained model doesn't give any results on the dataset aukerman.
"""

import os
from typing import Iterator

import cv2
import numpy as np
import torch
from tqdm import tqdm

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "aukerman")
PATCH_SIZE = 306
IMAGE_SIZE = (4896, 3672)

net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)


def load_images(folder_path: str) -> Iterator[np.ndarray]:
    for img_path in sorted(filter(lambda p: p.endswith(".JPG"), os.listdir(folder_path))):
        yield cv2.imread(os.path.join(folder_path, img_path))


def resize_images(images: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
    for img in images:
        yield cv2.resize(img, (IMAGE_SIZE[0] // 8, IMAGE_SIZE[1] // 8))


def split_image(image: np.ndarray, patch_size: int) -> Iterator[np.ndarray]:
    h, w, _ = image.shape
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            yield image[y:y + patch_size, x:x + patch_size]


if __name__ == "__main__":
    for img in tqdm(resize_images(load_images(DATASET_PATH))):
        # assert img.shape[0] % PATCH_SIZE == 0 and img.shape[1] % PATCH_SIZE == 0, \
        #     f"{img.shape} is not a multiple of {PATCH_SIZE}"

        # segmented_image = np.zeros(img.shape[:2], dtype=np.float32)

        # for y in range(0, img.shape[0], PATCH_SIZE):
        #     for x in range(0, img.shape[1], PATCH_SIZE):
        #         patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        #         if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
        #             continue  # Skip incomplete patches

        # segmented_patch = net(
        #     torch.tensor(patch.transpose(2, 0, 1)).unsqueeze(0).float() / 255).squeeze().detach().numpy()
        # segmented_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = segmented_patch[0, :, :]

        segmented_image = net(
            torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255).squeeze().detach().numpy()

        cv2.imshow("image", img)
        cv2.imshow("segmented_image", segmented_image[0, :, :])

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
