#!/usr/bin/env python3
"""

"""

import time

import cv2
import numpy as np

from scripts.matching.test_framework import benchmark

orb = cv2.ORB_create()
bf = cv2.BFMatcher()


def compare(_: int, im1: np.ndarray, im2: np.ndarray):
    t_detect = time.time_ns()
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    t_detect = time.time_ns() - t_detect

    t_match = time.time_ns()
    matches = bf.match(des1, des2)
    mkpts_0 = np.array([kp1[match.queryIdx].pt for match in matches])
    mkpts_1 = np.array([kp2[match.trainIdx].pt for match in matches])
    t_match = time.time_ns() - t_match

    t_score = time.time_ns()
    scores = np.array([m.distance for m in matches])
    t_score = time.time_ns() - t_score

    if scores.shape == (0,):
        stats_scores = (-1, -1, -1, -1)
    else:
        stats_scores = (np.min(scores), np.max(scores), np.mean(scores), np.std(scores))

    return (mkpts_0, mkpts_1,
            (t_match, t_detect, t_score,
             *stats_scores,
             mkpts_0.shape[0]))


if __name__ == "__main__":
    benchmark(compare,
              1 / 2,
              ["match duration", "detect duration", "score duration",
               "min distance", "max distance", "mean distance", "std distance",
               "number of matched point"],
              "orb")
