"""
Download tiles from IGN servers to reconstruct a map containing a square defined by the
top-left and bottom-right corners (GNSS coordinates).
"""

import math
from typing import Tuple, Optional

import cv2
import numpy as np
import requests
from tqdm import tqdm

TILE_SIZE = 256  # Pixels per tile
INITIAL_RESOLUTION = 2 * math.pi * 6378137 / TILE_SIZE  # Web Mercator resolution
ORIGIN_SHIFT = 2 * math.pi * 6378137 / 2.0  # Origin shift


def gps_to_wmts(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert GPS coordinates (lat, lon) to WMTS TILECOL and TILEROW."""
    # Convert lat/lon to Web Mercator (EPSG:3857)
    mx = lon * ORIGIN_SHIFT / 180.0
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * ORIGIN_SHIFT / 180.0

    # Calculate tile index
    res = INITIAL_RESOLUTION / (2 ** zoom)
    tile_col = int((mx + ORIGIN_SHIFT) / (res * TILE_SIZE))
    tile_row = int((ORIGIN_SHIFT - my) / (res * TILE_SIZE))

    return tile_col, tile_row


def get_tile(col: int, row: int, zoom: int) -> Optional[np.ndarray]:
    url = f"https://data.geopf.fr/wmts?REQUEST=GetTile&SERVICE=WMTS&VERSION=1.0.0&TILEMATRIXSET=PM&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg&TILECOL={col}&TILEROW={row}&TILEMATRIX={zoom}"
    response = requests.get(url)
    if response.status_code == 200:
        return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    return None


def get_map(top_left: Tuple[float, float], bot_right: Tuple[float, float], zoom: int = 18) -> Optional[np.ndarray]:
    tl_col, tl_row = gps_to_wmts(*top_left, zoom)
    br_col, br_row = gps_to_wmts(*bot_right, zoom)

    width = (br_col - tl_col + 1) * TILE_SIZE
    height = (br_row - tl_row + 1) * TILE_SIZE

    map_img = np.zeros((height, width, 3), dtype=np.uint8)

    for col in tqdm(range(tl_col, br_col + 1)):
        for row in tqdm(range(tl_row, br_row + 1), leave=False):
            tile_img = get_tile(col, row, zoom)
            if tile_img is not None:
                x_offset = (col - tl_col) * TILE_SIZE
                y_offset = (row - tl_row) * TILE_SIZE
                map_img[y_offset:y_offset + TILE_SIZE, x_offset:x_offset + TILE_SIZE] = tile_img

    return map_img


tr = (48.869344, 1.881983)
br = (48.852348, 1.9083)

map = get_map(tr, br)
cv2.imwrite("ign_map.jpg", map)

# latitude = 48.864561
# longitude = 1.892660
