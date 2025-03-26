"""
This is meant to test xfeat, xfeat* and xfeat+lighterglue, omniglue and other image matching techniques.
"""

import argparse
import time
from pathlib import Path
from typing import Callable, Tuple, Union, Literal

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from tqdm import tqdm

from scripts.matching.generate_test_data import corr_dataset
from scripts.utils.datasets import get_dataset_by_name, resize_image
from scripts.utils.monitoring import CPURamMonitor


def get_homography_results(ref_points, dst_points, ugv_image, uav_image):
    """
    Given a list of ref and dst points, compute a homography and the region in common with both images
    :param ref_points: the reference points
    :param dst_points: the destination points of those reference points
    :param ugv_image: the reference image (ugv image)
    :param uav_image: the destination image (uav image)
    :return: (the corners of the common region in the uav image, those points in the ugv image,
                the common region in the uav image, those points in the uav image,
                the mask computed with the homography)
    """

    ugv_h, ugv_w = ugv_image.shape[:2]
    uav_h, uav_w = uav_image.shape[:2]
    # found homography given the matched points
    M, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    if M is None:
        return None, None, None, None, None
    # find the ugv image corners in the uav image
    ugv_corners_in_uav = cv2.perspectiveTransform(
        np.array([[0, 0], [ugv_w - 1, 0], [ugv_w - 1, ugv_h - 1], [0, ugv_h - 1]],
                 dtype=np.float32).reshape(-1, 1, 2), M)
    # clip them so that there are in the uav images
    ugv_corners_in_uav = np.clip(ugv_corners_in_uav, (0, 0), (uav_w - 1, uav_h - 1))
    # found them back in the ugv image
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None, None, None, None, None
    ugv_corners_clipped = np.clip(cv2.perspectiveTransform(ugv_corners_in_uav, M_inv),
                                  (0, 0), (ugv_w - 1, ugv_h - 1))
    # found a close enough rectangle (the closer, the fewer deformations)
    min_pt, max_pt = (np.min(ugv_corners_clipped.reshape(-1, 2), axis=0),
                      np.max(ugv_corners_clipped.reshape(-1, 2), axis=0))
    ugv_rect_size = (int(max_pt[0] - min_pt[0]), int(max_pt[1] - min_pt[1]))
    ugv_rect_corners = np.array([[0, 0], [ugv_rect_size[0], 0],
                                 [ugv_rect_size[0], ugv_rect_size[1]], [0, ugv_rect_size[1]]], dtype=np.float32)
    # extract the common region in both images
    uav_m = cv2.getPerspectiveTransform(ugv_corners_in_uav.astype(np.float32), ugv_rect_corners)
    ugv_m = cv2.getPerspectiveTransform(ugv_corners_clipped.astype(np.float32), ugv_rect_corners)
    uav_warped_region = cv2.warpPerspective(uav_image, uav_m, ugv_rect_size)
    ugv_warped_region = cv2.warpPerspective(ugv_image, ugv_m, ugv_rect_size)
    return (ugv_corners_in_uav, ugv_corners_clipped,
            normalize_image(uav_warped_region), normalize_image(ugv_warped_region), mask)


def cross_detect(points: np.ndarray) -> bool:
    p1, p2, p3, p4 = points.reshape(4, 2)
    return (np.cross(p3 - p1, p2 - p1) * np.cross(p4 - p1, p3 - p1) < 0) and (
            np.cross(p4 - p2, p1 - p2) * np.cross(p3 - p2, p4 - p2) < 0)


def normalize_image(image: np.ndarray) -> np.ndarray:
    return ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)


def get_matching_scores(ref_points, dst_points, ugv_image, uav_image):
    """
    Given a list of ref and dst points, compute a matching score
    :param ref_points: the reference points
    :param dst_points: the destination points of those reference points
    :param ugv_image: the reference image (ugv image)
    :param uav_image: the destination image (uav image)
    :return: (area of the common region in the uav image, area of the common region in the ugv image,
                mean squared error, structural_similarity)
    """

    if ref_points.shape[0] < 4:
        return [-1] * 8

    (ugv_corners_in_uav, ugv_corners_clipped,
     uav_warped_region, ugv_warped_region, _) = get_homography_results(ref_points, dst_points,
                                                                       ugv_image, uav_image)
    if ugv_corners_in_uav is None:
        return [-1] * 8
    is_crossed_uav = cross_detect(ugv_corners_in_uav)
    is_crossed_ugv = cross_detect(ugv_corners_clipped)
    count_black_pixels_uav = np.sum(np.all(uav_warped_region == [0, 0, 0], axis=-1))
    count_black_pixels_ugv = np.sum(np.all(ugv_warped_region == [0, 0, 0], axis=-1))
    area_uav = cv2.contourArea(ugv_corners_in_uav.astype(np.float32))
    area_ugv = cv2.contourArea(ugv_corners_clipped.astype(np.float32))

    try:
        mse = mean_squared_error(ugv_warped_region, uav_warped_region)
    except:
        mse = -1

    try:
        ssim, _ = structural_similarity(ugv_warped_region, uav_warped_region,
                                        multichannel=True, full=True, data_range=255, channel_axis=2)
    except:
        ssim = -1

    # TODO mutual information
    return (area_uav, area_ugv,
            count_black_pixels_uav, count_black_pixels_ugv,
            is_crossed_uav, is_crossed_ugv,
            mse, ssim)


def serialize_nparray(np_array):
    """
    :param np_array: the numpy array to be serialized
    :return: the serialized numpy array to be saved in a csv column
    """
    return ';'.join(' '.join(map(str, row)) for row in np_array)


def deserialize_nparray(serialized_str):
    """
    :param serialized_str: the serialized numpy array to be loaded
    :return: the deserialized numpy array
    """
    return np.array([list(map(float, row.split(' '))) for row in serialized_str.split(';')])


def rotate_image(image, angle: Union[Literal[0], Literal[90], Literal[180], Literal[270]]):
    """
    :param image: an image
    :param angle: a rotation angle (0, 90, 180, 270)
    :return: the rotated image
    """
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def benchmark(compare: Callable[[int, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]],
              compare_duration_estimate: float,
              metrics_name: list[str],
              output_name: str):
    """
    A general purpose benchmark function. It is supposed to be called to generate some csv

    :param compare: The function that is supposed to be compared two images and return a bench of metrics
    :param compare_duration_estimate: the estimate of the duration (s) for the comparison (in order to limit the benchmark to 1h).
    :param metrics_name: the name of the metrics returned by the compare_lg function
    :param output_name: the name of the file to save the results (path is hard-coded)
    """
    uav_location = pd.read_csv(get_dataset_by_name(corr_dataset / "uav_patch" / "location.csv"))
    ugv_location = pd.read_csv(get_dataset_by_name(corr_dataset / "ugv_bev_patch" / "location.csv"))

    ordered_idx = []

    for uav in tqdm(uav_location.itertuples()):
        uav_coord = (uav.x_gnss, uav.y_gnss, uav.z_gnss)
        uav_angle = (uav.roll_gnss, uav.pitch_gnss, uav.yaw_gnss)

        ugv_idx = (ugv_location['timestamp'] - uav.timestamp).abs().argmin()
        ugv = ugv_location.iloc[ugv_idx]

        ugv_coord = (ugv.x_gnss, ugv.y_gnss, ugv.z_gnss)
        ugv_angle = (ugv.roll_gnss, ugv.pitch_gnss, ugv.yaw_gnss)
        dist = np.linalg.norm(np.array(uav_coord) - np.array(ugv_coord))
        angle_dist = np.linalg.norm(np.array(uav_angle) - np.array(ugv_angle))

        ordered_idx.append((uav.Index, ugv_idx, dist, angle_dist))

    ordered_idx = (pd.DataFrame(ordered_idx,
                                columns=['uav_index', 'ugv_index', 'distance', 'angle_distance'])
                   # .sort_values(by=['distance', 'angle_distance'])
                   )

    # try to limit benchmark at 30min
    n = int(30 * 60 / 8 / compare_duration_estimate)

    results = []

    for i in tqdm(range(0, len(ordered_idx), len(ordered_idx) // n)):
        uav_idx = int(ordered_idx.iloc[i]['uav_index'])
        ugv_idx = int(ordered_idx.iloc[i]['ugv_index'])
        dist = ordered_idx.iloc[i]['distance']
        angle_dist = ordered_idx.iloc[i]['angle_distance']

        uav_image_path = list(Path(get_dataset_by_name(corr_dataset / "uav_patch")).glob(
            f"{int(uav_location.iloc[uav_idx]['timestamp'])}_*.*"))[0]
        ugv_image_path = list(Path(get_dataset_by_name(corr_dataset / "ugv_patch")).glob(
            f"{int(ugv_location.iloc[ugv_idx]['timestamp'])}_*.*"))[0]
        ugv_bev_image_path = list(Path(get_dataset_by_name(corr_dataset / "ugv_bev_patch")).glob(
            f"{int(ugv_location.iloc[ugv_idx]['timestamp'])}_*.*"))[0]

        def call_compare(uav_path, ugv_path, is_bev: bool,
                         angle: Union[Literal[0], Literal[90], Literal[180], Literal[270]]):
            uav_image = rotate_image(cv2.imread(uav_path), angle)
            ugv_image = cv2.imread(ugv_path)
            ram = CPURamMonitor()
            ram.start()
            t = time.time_ns()
            ref_points, dst_points, (cmp_metrics) = compare(i, ugv_image, uav_image)
            t = time.time_ns() - t
            ram.stop()

            scores = get_matching_scores(ref_points, dst_points, ugv_image, uav_image)
            results.append((uav_path, ugv_path, angle, is_bev,
                            uav_idx, ugv_idx,
                            dist, angle_dist,
                            serialize_nparray(ref_points), serialize_nparray(dst_points),
                            *scores,
                            t, ram.get_average_ram(), ram.get_max_ram(),
                            *cmp_metrics))

        for ang in (0, 90, 180, 270):
            call_compare(uav_image_path, ugv_bev_image_path, True, ang)
            call_compare(uav_image_path, ugv_image_path, False, ang)

    pd.DataFrame(results, columns=['uav_path', 'ugv_path', 'uav_image_rotation', 'ugv image is bev',
                                   'uav_idx location.csv', 'ugv_idx in location.csv',
                                   'dist', 'angle_dist',
                                   'ref_points', 'dst_points',
                                   'area_uav', 'area_ugv',
                                   'count_black_pixels_uav', 'count_black_pixels_ugv',
                                   'is_crossed_uav', 'is_crossed_ugv',
                                   'mse', 'ssim',
                                   'compare_duration', 'compare_average_ram', 'compare_max_ram',
                                   *metrics_name]).to_csv(get_dataset_by_name(corr_dataset) + f'/{output_name}.csv',
                                                          index=True)


def plot_corners(img, corners):
    """
    Plot corners on an img
    :param img: the image
    :param corners: the corners
    :return: the image with lines to show the corners
    """
    colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    img_with_corners = img.copy()
    corners_r = corners.reshape(-1, 1, 2)
    for i in range(len(corners_r)):
        start_point = tuple(corners_r[i - 1][0].astype(int))
        end_point = tuple(corners_r[i][0].astype(int))
        cv2.line(img_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners
        cv2.circle(img_with_corners, end_point, 5, colors[i], 3)
    return img_with_corners


def show_results(filename, path_relative_transform=None):
    """
    Shows the results of the matching for all the tested images
    :param filename: the csv file
    """
    if path_relative_transform is None:
        path_relative_transform = lambda x: x

    df = pd.read_csv(filename)
    # plot data
    from scripts.utils.plot import plot_histogram
    import matplotlib.pyplot as plt

    df['valid'] = ((df['area_uav'] > 10000) & (df['area_ugv'] > 10000)
                   & (df['count_black_pixels_uav'] < 20) & (df['count_black_pixels_ugv'] < 20)
                   & (df['is_crossed_uav'] == 'False') & (df['is_crossed_ugv'] == 'False'))

    print(f"{int(df['valid'].mean() * 100)}% valid points.")

    plot_histogram(df.loc[df['valid'], 'ssim'].values, bins=50, title="Histogram of SSIM (Valid == True)")
    plot_histogram(df.loc[~df['valid'], 'ssim'].values, bins=50, title="Histogram of SSIM (Valid == False)")

    grouped = df[df['valid']].groupby('uav_image_rotation')
    for rotation, group in grouped:
        fig, ax1 = plt.subplots()
        ax1.scatter(group['number of matched point'], group['ssim'], color='b', label='SSIM', alpha=0.7)
        ax1.set_xlabel('Number of Matched Points')
        ax1.set_ylabel('SSIM', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.scatter(group['number of matched point'], group['mse'], color='r', label='MSE', alpha=0.7)
        ax2.set_ylabel('MSE', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title(f'SSIM vs MSE vs Number of Matched Points (Rotation: {rotation})')
        plt.tight_layout()
        plt.show()

    # show results in open cv
    is_running = False

    for _, row in tqdm(df.iterrows(), total=len(df)):
        uav_image_path = path_relative_transform(row["uav_path"])
        ugv_image_path = path_relative_transform(row["ugv_path"])
        mse = row["mse"]
        ssim = row["ssim"]
        is_bev = row["ugv image is bev"]
        angle = row["uav_image_rotation"]
        uav_image = resize_image(rotate_image(cv2.imread(uav_image_path), int(angle)), 400, 400)
        ugv_image = resize_image(cv2.imread(ugv_image_path), 400, 400)

        if not row["valid"]:
            continue

        ref_points = deserialize_nparray(row["ref_points"])
        dst_points = deserialize_nparray(row["dst_points"])

        if ref_points.shape[0] > 4:
            (ugv_corners_in_uav, ugv_corners_clipped,
             uav_warped_region, ugv_warped_region, mask) = get_homography_results(ref_points, dst_points,
                                                                                  ugv_image, uav_image)

            if ugv_corners_in_uav is not None:
                img1_with_corners = plot_corners(ugv_image, ugv_corners_clipped)
                img2_with_corners = plot_corners(uav_image, ugv_corners_in_uav)
            else:
                img1_with_corners, img2_with_corners = ugv_image, uav_image
                uav_warped_region, ugv_warped_region = ugv_image, uav_image
        else:
            img1_with_corners, img2_with_corners = ugv_image, uav_image
            uav_warped_region, ugv_warped_region = ugv_image, uav_image

        keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
        keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

        img_with_corners = cv2.drawMatches(img1_with_corners, [], img2_with_corners, [],
                                           [], None, flags=2)
        img_warped = cv2.drawMatches(ugv_warped_region, [], uav_warped_region, [],
                                     [], None, flags=2)
        if ref_points.shape[0] > 4:
            img_with_matches = cv2.drawMatches(ugv_image, keypoints1, uav_image, keypoints2, matches, None, flags=2)
        else:
            img_with_matches = img_with_corners

        cv2.putText(img_with_corners,
                    f"ugv {'bev ' if is_bev else ''}vs uav at {angle}deg mse {mse:.0f} ssim {ssim:.1f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)

        cv2.imshow("img_with_corners", img_with_corners)
        cv2.imshow("img_with_matches", img_with_matches)
        cv2.imshow("img_warped", img_warped)

        k = cv2.waitKey(100 if is_running else 0) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('p'):
            is_running = not is_running


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b", action='store_true', default=False,
                        help="Run test of the benchmark")
    parser.add_argument("--viz", "-v", default="", type=Path,
                        help="The path to the result file to visualize")
    args = parser.parse_args()

    if args.benchmark:
        def dummy(_, __, ___):
            return ()


        benchmark(dummy, 5, [], "a")
    elif args.viz:
        show_results(args.viz, lambda x: x.replace("/app/", "/home/seb-sti1/sebspace/"))
    else:
        print("You need to specify what to do")
