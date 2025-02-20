"""
This uses detectree model to segment IGN maps
"""

import cv2

from scripts.terrain_segmentation.detectree_test import make_predictions
from scripts.terrain_segmentation.get_ign_map import get_map


def main():
    tr = (48.869344, 1.881983)
    br = (48.852348, 1.9083)

    map = get_map(tr, br, transform=make_predictions)
    cv2.imwrite("ign_map_classified.jpg", map)


if __name__ == "__main__":
    main()
