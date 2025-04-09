import argparse
import os.path
from pathlib import Path
from typing import Optional, Literal, Tuple, List

import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.utils.datasets import get_dataset_by_name, resize_image, load_paths_and_files
from scripts.utils.norlab_sync_viz import seq1

corr_dataset = Path("norlab_ulaval_datasets/correspondence_dataset")


def get_aruco(img: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if len(corners) == 0:
        return None
    return corners[0]


def create_matrix_front_patch_using_aruco(aruco_square: np.ndarray,
                                          square_size: float,
                                          pattern: Tuple[int, int] = (1, 1),
                                          margin: Tuple[float, float] = (0, 0),
                                          aruco_size: float = 0.35) -> List[np.ndarray]:
    """
    Creates a matrix of patches in front of the robot detected using ArUco markers.

    :param aruco_square: A NumPy array representing the ArUco marker square.
    :param square_size: The size of each square in meters.
    :param pattern: A tuple (rows, cols) defining the grid pattern for the patches.
    :param margin: A tuple (x_margin, y_margin) specifying the spacing between patches in meters.
    :param aruco_size: The size of the ArUco marker within each square in meters.
    :return: A list of NumPy arrays, each representing a patch with an ArUco marker.
    """
    aruco_square = aruco_square.astype(np.float64)
    center = np.mean(aruco_square, axis=0)
    vectors = aruco_square - center
    scale_factor = square_size / aruco_size
    # the rotation is because of the orientation of the aruco on the Warthog in the real world
    new_corners = center + (vectors @ np.array([[0., 1.], [-1., 0.]])) * scale_factor

    pixel_by_meters = np.linalg.norm(vectors[0, :] - vectors[1, :]) / aruco_size
    right = aruco_square[2] - aruco_square[1]
    right *= pixel_by_meters * (square_size + margin[0]) / np.linalg.norm(right)
    forward = aruco_square[0] - aruco_square[1]
    forward *= - pixel_by_meters * (square_size + margin[1]) / np.linalg.norm(forward)

    squares = []
    for i in range(pattern[0]):
        for j in range(pattern[1]):
            c = (new_corners + forward * (i + 1) + right * (j - (pattern[1] - 1) / 2))

            if -1 < (j - (pattern[1] - 1) / 2) < 1:  # avoid robot's shadows
                c += forward * 0.25

            squares.append(c.astype(np.uint))

    return squares


def create_matrix_front_patch(square_size: float,
                              image_size: Tuple[int, int],
                              pattern: Tuple[int, int] = (1, 1),
                              margin: Tuple[float, float] = (0, 300)) -> List[np.ndarray]:
    """
    Creates a matrix of squares with a given size in pixels, aligned with the image
    and centered at the bottom of the image.

    :param square_size: The size of each square in pixels.
    :param image_size: The size of the image in pixels.
    :param pattern: The pattern of the matrix (rows, columns).
    :param margin: The margin to be applied in pixels to each square.
    :return: A list of 2D arrays, each representing the corners of a square in pixels.
    """
    squares = []
    for i in range(pattern[0]):
        for j in range(pattern[1]):
            c = np.array([image_size[1] / 2 + (square_size + margin[0]) * (j - (pattern[1] - 1) / 2),
                          image_size[0] - square_size / 2 - margin[1] + (- square_size - margin[1]) * i],
                         dtype=np.float64)

            corners = np.array([
                c + np.array([-square_size / 2, -square_size / 2]),  # top-left
                c + np.array([square_size / 2, -square_size / 2]),  # top-right
                c + np.array([square_size / 2, square_size / 2]),  # bottom-right
                c + np.array([-square_size / 2, square_size / 2])  # bottom-left
            ], dtype=np.float64)

            squares.append(corners.astype(np.uint))
    return squares


def apply_rotations(corners: List[np.ndarray], angles: List[float]) -> List[np.ndarray]:
    rotated_rects = []
    for corner_set in corners:
        center = np.mean(corner_set, axis=0)
        for angle in angles:
            theta = np.radians(-angle)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            rotated_corners = np.dot(corner_set - center, rotation_matrix.T) + center
            rotated_rects.append(rotated_corners.astype(np.int32))
    return rotated_rects


def extract_patch(corners: np.ndarray, src_image: np.ndarray, side_width: int) -> np.ndarray:
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), np.array(
        [[0, 0], [side_width - 1, 0], [side_width - 1, side_width - 1], [0, side_width - 1]],
        dtype=np.float32))
    return cv2.warpPerspective(src_image, M, (side_width, side_width))


def generate_patch(patch_type: Literal["uav", "ugv", "ugv_bev"], vis=False):
    out_dir = Path(get_dataset_by_name(os.path.join(corr_dataset, f"{patch_type}_patch")))

    gnss = pd.read_csv(get_dataset_by_name(seq1 / "rtk_odom/rtk_odom.csv"))
    wheel = pd.read_csv(get_dataset_by_name(seq1 / "wheel_odom/wheel_odom.csv"))

    merged_data = pd.DataFrame()
    angles = [-10, -5, 0, 5, 10]
    is_running = True

    for filepath, image in tqdm(load_paths_and_files(
            get_dataset_by_name(seq1 / ("aerial" if patch_type == "uav" else "ground") /
                                ("projections" if patch_type == "ugv_bev" else "images")))):
        uav_time = int(filepath.stem)
        # find aruco
        if patch_type == "uav":
            aruco_loc = get_aruco(image)
            if aruco_loc is None:
                continue
            aruco_loc = get_aruco(image)[0, :, :]
            c = create_matrix_front_patch_using_aruco(aruco_loc, 2)
            c = apply_rotations(c, angles)
        else:
            c = create_matrix_front_patch(420 if patch_type == "ugv" else 500, image.shape)

        # extract image patchs
        for index, square in enumerate(c):
            if not (0 <= square[0][0] < image.shape[1] and 0 <= square[0][1] < image.shape[0] and
                    0 <= square[1][0] < image.shape[1] and 0 <= square[1][1] < image.shape[0] and
                    0 <= square[2][0] < image.shape[1] and 0 <= square[2][1] < image.shape[0] and
                    0 <= square[3][0] < image.shape[1] and 0 <= square[3][1] < image.shape[0]):
                continue
            side_length = int(np.linalg.norm(square.astype(np.float32)[0] - square[1]))
            if 50 > side_length or side_length > 1000:
                continue
            extracted_square = extract_patch(square, image, side_length)

            savepath = str(out_dir / f"{filepath.stem}_{index}{filepath.suffix}")
            if not vis:
                cv2.imwrite(savepath, extracted_square)

            search_gnss = gnss.iloc[(gnss['timestamp'] - uav_time).abs().argmin()]
            search_wheel = wheel.iloc[(wheel['timestamp'] - uav_time).abs().argmin()]
            temp_data = pd.DataFrame(pd.concat([
                search_gnss[['timestamp', 'ros_time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']].add_suffix('_gnss'),
                search_wheel[['timestamp', 'ros_time', 'vel_x', 'vel_y', 'vel_z']].add_suffix('_wheel')
            ], axis=0)).T
            temp_data["timestamp"] = uav_time
            temp_data["filepath"] = savepath
            temp_data["angle"] = 0 if patch_type != "uav" else angles[index]
            merged_data = pd.concat([merged_data, temp_data], axis=0).reset_index(drop=True)

        if vis:
            cv2.polylines(image, c, isClosed=True, color=(0, 255, 0), thickness=4)
            cv2.imshow("image", resize_image(image, 600))
            k = cv2.waitKey(5 if is_running else 0) & 0xFF
            if k == ord('q'):
                break
            elif k == ord(' '):
                is_running = not is_running

    if not vis:
        merged_data.to_csv(out_dir / "location.csv", index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process UAV and/or UGV data to create a dataset"
                                                 "of patches to be used to find correspondences.")
    parser.add_argument("--type", choices=["uav", "ugv", "ugv_bev"], action="append",
                        help="Specify data type(s) to process. By default, it does both (equivalent to --type uav --type ugv).")
    parser.add_argument("--display", action="store_true", help="Enable display output (default: False).")
    args = parser.parse_args()

    if args.type is None:
        args.type = ["uav", "ugv", "ugv_bev"]

    if "uav" in args.type:
        generate_patch("uav", vis=args.display)
    if "ugv" in args.type:
        generate_patch("ugv", vis=args.display)
    if "ugv_bev" in args.type:
        generate_patch("ugv_bev", vis=args.display)
