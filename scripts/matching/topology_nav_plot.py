import logging
from typing import List

import cv2
import numpy as np

from scripts.matching.topology_nav_graph import Node

logger = logging.getLogger(__name__)


def generate_match_grid(n: Node,
                        ugv_c: np.ndarray,
                        extracted_patches: List[np.ndarray]) -> np.ndarray:
    grid_images = []
    xfeat = get_xfeat()

    for img_path, uav_c, uav_feature in n.correspondance_data:
        row_images = []
        full_img = cv2.imread(img_path)

        for extracted_patch in extracted_patches:
            ugv_feature = xfeat.detectAndCompute(extracted_patch, top_k=4096)[0]
            ugv_feature.update({'image_size': (extracted_patch.shape[1], extracted_patch.shape[0])})
            mkpts_0, mkpts_1, _ = xfeat.match_lighterglue(ugv_feature, uav_feature)

            if mkpts_0.shape[0] == 0:
                h1, w1 = extracted_patch.shape[:2]
                h2, w2 = full_img.shape[:2]
                match_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            else:
                kp1 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in mkpts_0]
                kp2 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in mkpts_1]
                matches = [cv2.DMatch(i, i, 0) for i in range(len(mkpts_0))]

                match_img = cv2.drawMatches(
                    extracted_patch, kp1,
                    full_img, kp2,
                    matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
            text = f'{mkpts_0.shape[0]} matches. {uav_c[0]:.1f}, {uav_c[1]:.1f}. {np.linalg.norm(uav_c - ugv_c):.1f}'
            cv2.putText(match_img, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

            row_images.append(match_img)
        grid_images.append(np.hstack(row_images))
    return np.vstack(grid_images)
