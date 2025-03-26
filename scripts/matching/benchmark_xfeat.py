"""
This was used to test xfeat+lighterglue.py.

Usage (be aware that there might some missing pip install instructions):
1. Clone https://github.com/verlab/accelerated_features
2. python3.11 -m venv .venv
3. source .venv/bin/activate
4. pip install torch==torch==2.0.0-1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
5. pip install opencv-contrib-python tqdm numpy
6. Place this script in root folder of the xfeat repo
7. change package_folder so that it corresponds to the location of this repo https://github.com/Seb-sti1/mastersthesis
8. Run this file

"""
import os
import sys
import time

import numpy as np
import torch

# from modules.xfeat import XFeat
print(f"Cuda device(s) {os.environ['CUDA_VISIBLE_DEVICES']}"
      if "CUDA_VISIBLE_DEVICES" in os.environ else "No CUDA devices")
# os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)

package_folder = '/app/mastersthesis/'
sys.path.append(package_folder)

from scripts.matching.test_framework import benchmark


def compare_lg(i: int, im1: np.ndarray, im2: np.ndarray):
    im1 = np.copy(im1)
    im2 = np.copy(im2)

    t_detect = time.time_ns()
    # Inference with batch = 1
    output0 = xfeat.detectAndCompute(im1, top_k=4096)[0]
    output1 = xfeat.detectAndCompute(im2, top_k=4096)[0]
    t_detect = time.time_ns() - t_detect

    t_update = time.time_ns()
    # Update with image resolution (required)
    output0.update({'image_size': (im1.shape[1], im1.shape[0])})
    output1.update({'image_size': (im2.shape[1], im2.shape[0])})
    t_update = time.time_ns() - t_update

    t_match = time.time_ns()
    mkpts_0, mkpts_1, _ = xfeat.match_lighterglue(output0, output1)
    t_match = time.time_ns() - t_match

    o0s_np = output0['scores'].cpu().numpy()
    o1s_np = output1['scores'].cpu().numpy()

    return (mkpts_0, mkpts_1,
            (t_detect, t_update, t_match,
             np.min(o0s_np), np.max(o0s_np), np.mean(o0s_np), np.std(o0s_np),
             np.min(o1s_np), np.max(o1s_np), np.mean(o1s_np), np.std(o1s_np),
             mkpts_0.shape[0]))


def compare(i: int, im1: np.ndarray, im2: np.ndarray):
    im1 = np.copy(im1)
    im2 = np.copy(im2)

    t_match = time.time_ns()
    mkpts_0, mkpts_1 = xfeat.match_xfeat(im1, im2)
    t_match = time.time_ns() - t_match

    return (mkpts_0, mkpts_1,
            (t_match, mkpts_0.shape[0]))


def compare_star(i: int, im1: np.ndarray, im2: np.ndarray):
    im1 = np.copy(im1)
    im2 = np.copy(im2)

    t_match = time.time_ns()
    mkpts_0, mkpts_1 = xfeat.match_xfeat_star(im1, im2)
    t_match = time.time_ns() - t_match

    return (mkpts_0, mkpts_1,
            (t_match, mkpts_0.shape[0]))


if __name__ == '__main__':
    benchmark(compare_lg,
              0.5,
              ["detect duration", "update duration", "match duration",
               "min im0 scores", "max im0 scores", "mean im0 scores", "std im0 scores",
               "min im1 scores", "max im1 scores", "mean im1 scores", "std im1 scores",
               "number of matched point"],
              "xfeat+lighterglue")

    benchmark(compare,
              0.5,
              ["match duration",
               "number of matched point"],
              "xfeat")

    benchmark(compare_star,
              0.5,
              ["match duration",
               "number of matched point"],
              "xfeat+star")
