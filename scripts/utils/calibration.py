"""
This file deals with calibration of cameras. It was used for the calibration of the Viewpro Q10F.

A helpful comment to do better calibration
https://stackoverflow.com/questions/12794876/how-to-verify-the-correctness-of-calibration-of-a-webcam/12821056#12821056

I had some issues with some weird distortion when not able to have chessboard image on the whole image
https://stackoverflow.com/questions/40775102/calibrate-camera-with-only-radial-distortion
"""
import argparse
import logging
from typing import Iterator, Tuple, Optional

import cv2
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from scripts.utils.datasets import load_files, get_dataset_by_name
from scripts.utils.plot import plot_bar, plot

logger = logging.getLogger(__name__)


def find_chessboard(pattern_size: Tuple[int, int], image: np.ndarray, resize: float = 2) -> Optional[np.ndarray]:
    """
    Find the chessboard in the image

    For far away chessboard, increasing the resize value can help. A good way to debug detection issue is to use
    the cv2.adaptiveThreshold and verify that each black square becomes a clear border (with a white "square" in
    the center).

    :param pattern_size: The pattern_size of the chessboard
    :param image: The image to find the chessboard in
    :param resize: The factor by which to resize the image (for far away chessboard it helps the detection)
    :return: The corners of the chessboard or None if no chessboard was found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=resize, fy=resize, interpolation=cv2.INTER_CUBIC)

    cv2.destroyAllWindows()
    ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners2 / resize
    return None


def find_correspondences(pattern_size: Tuple[int, int],
                         square_size: float,
                         images: Iterator[np.ndarray],
                         manual_validation: bool = True) -> Tuple[list[np.ndarray], list[np.ndarray], Tuple[int, int]]:
    """
    Create list of object and image points for an image iterator

    :param pattern_size: The size of the chessboard pattern
    :param square_size: The size (in mm) of each chessboard square
    :param images: An iterable of images to find the chessboard in
    :param manual_validation: If the user needs to validate each chessboard found
    :return: obj_points, img_points, (h, w)
    """
    h, w = None, None

    # prepare the pattern coordinates
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    # find the pattern in the world
    obj_points = []
    img_points = []
    for image in tqdm(images):
        h, w = image.shape[:2]
        corners = find_chessboard(pattern_size, image)

        if corners is None:
            continue

        for i in range(corners.shape[0]):
            corners[i, 0, :] += np.random.normal(0, 1, 2)

        use_image = True
        if not manual_validation:
            logger.info("Add image to calibration? (y/n)")
            cv2.drawChessboardCorners(image, pattern_size, corners, True)
            cv2.imshow('Preview', image)
            s = cv2.waitKey(0)
            if s != ord('y'):
                use_image = False

        if use_image:
            img_points.append(corners)
            obj_points.append(pattern_points)

    return obj_points, img_points, (h, w)


def calibrate(image_size: Tuple[int, int],
              obj_points: list[np.ndarray], img_points: list[np.ndarray],
              criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 2e-16),
              flag=cv2.CALIB_FIX_K3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibrate a camera for the intrinsics parameters

    :param image_size: the image sizes (h, w)
    :param obj_points: the coordinates of the chessboard corners in the world
    :param img_points: the coordinates of the chessboard corners in the image
    :param flag: cv2.CALIB_ZERO_TANGENT_DIST cv2.CALIB_FIX_K3
    :param criteria:
    :return: the camera matrix, the distortion matrix, the reprojection errors
    """
    h, w = image_size

    # initial values
    camera_matrix = np.array([[1000.0, 0, image_size[0] // 2],
                              [0, 1000.0, image_size[1] // 2],
                              [0, 0, 1]], np.float32)
    dist = np.array([0, 0, 0, 0, 0], np.float32)

    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h),
                                                                 camera_matrix, dist,
                                                                 None, None,
                                                                 flag, criteria=criteria)
    assert ret, "Calibration failed"

    reproject_errors = np.zeros((len(obj_points)))
    for i, (objp, imgp, rvec, tvec) in enumerate(zip(obj_points, img_points, rvecs, tvecs)):
        imgp_r, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist)
        reproject_errors[i] = cv2.norm(imgp, imgp_r)
    return camera_matrix, dist, reproject_errors


def filter_bad_calibration_images(obj_points: list[np.ndarray], img_points: list[np.ndarray],
                                  reproject_errors: np.ndarray,
                                  threshold: Optional[float] = None) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Filter bad calibration images based on reprojection errors
    :param obj_points: Object coordinates of the chessboard corners in the world
    :param img_points: Image coordinates of the chessboard corners in the image
    :param reproject_errors: The reprojection errors of the chessboard corners in the image
    :param threshold: The threshold to use for filtering
    :return: new_obj_points, new_img_points, new_reproject_errors
    """
    if threshold is None:
        return obj_points, img_points
    new_obj_points = []
    new_img_points = []
    new_reproject_errors = []
    for objp, imgp, reproject_error in zip(obj_points, img_points, reproject_errors):
        if reproject_error < threshold:
            new_obj_points.append(objp)
            new_img_points.append(imgp)
            new_reproject_errors.append(reproject_error)
    return new_obj_points, new_img_points


def calibration_comparison(calibration_sets: list[str], automatic: bool = True,
                           threshold: Optional[float] = 15.):
    """
    Create calibration intrinsics parameters for a list of calibration sets
    :param calibration_sets: Folder names (placed in datasets/viewpro_calibration/)
    :param automatic: If the user needs to validate each chessboard found
    :param threshold: The threshold to use for filtering
    """

    camera_parameters = {}
    errors = {}
    distortions = {}
    obj_points = {}
    img_points = {}

    for calibration_set in calibration_sets:
        obj_pts, img_pts, image_sizes = find_correspondences((8, 6), 0.071,
                                                             load_files(get_dataset_by_name(
                                                                 f"viewpro_calibration/{calibration_set}")),
                                                             automatic)
        camera_matrix, dist, reproject_error = calibrate(image_sizes, obj_pts, img_pts)
        obj_pts, img_pts = filter_bad_calibration_images(obj_pts, img_pts, reproject_error, threshold)
        camera_matrix, dist, reproject_error = calibrate(image_sizes, obj_pts, img_pts)

        if camera_matrix is None:
            continue
        camera_parameters[calibration_set] = [camera_matrix[0, 0], camera_matrix[1, 1],
                                              camera_matrix[0, 2], camera_matrix[1, 2]]
        errors[calibration_set] = [np.average(reproject_error), np.max(reproject_error)]
        distortions[calibration_set] = dist[:, 0]
        obj_points[calibration_set] = obj_pts
        img_points[calibration_set] = img_pts

    plot_bar(["fx", "fy", "cx", "cy"], camera_parameters,
             title="Calibration results")
    plot_bar(["avg error", "max error"], errors,
             title="Reprojection error")
    plot_bar(["k1", "k2", "p1", "p2", "k3"], distortions,
             title="Distortion coefficient")


def full_calibration():
    """
    Use all calibration images in focus2* to create a calibration matrix.
    To validate the calibration, it, after excluding outliers, does the calibration using 5-Fold. For each fold,
     it calibrates using 80% of the images and computes the reprojection errors on the remaining 20%.

    :return: The calibration matrix using all the calibration images (except outliers),
                        the reprojection errors of each image for each fold.
    """

    def image_iter() -> Iterator[np.ndarray]:
        for calibration_set in ["focus2dist1_5-3", "focus2dist1_6-3", "focus2dist2_5-3", "focus2dist2_6-3"]:
            for image in load_files(get_dataset_by_name(f"viewpro_calibration/{calibration_set}")):
                yield image

    obj_pts, img_pts, image_sizes = find_correspondences((8, 6), 0.071,
                                                         image_iter(),
                                                         True)

    # Do a first calibration to remove outliers
    camera_matrix, dist, reproject_errors = calibrate(image_sizes, obj_pts, img_pts)
    obj_pts, img_pts = filter_bad_calibration_images(obj_pts, img_pts, reproject_errors, 11)
    obj_pts_ndarray, img_pts_ndarray = np.array(obj_pts), np.array(img_pts)

    reproject_errors = []

    for i, (train_idx, test_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(obj_pts)):
        obj_pts_train = obj_pts_ndarray[train_idx]
        obj_pts_test = obj_pts_ndarray[test_idx]
        img_pts_train = img_pts_ndarray[train_idx]
        img_pts_test = img_pts_ndarray[test_idx]

        camera_matrix, dist, _ = calibrate(image_sizes, obj_pts_train, img_pts_train)

        for objp, imgp in zip(obj_pts_test, img_pts_test):
            ret, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist)
            imgp_r, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist)
            reproject_errors.append(cv2.norm(imgp, imgp_r))

    camera_matrix, dist, _ = calibrate(image_sizes, obj_pts, img_pts)

    logger.info(f"fx {camera_matrix[0, 0]} fy {camera_matrix[1, 1]}"
                f" cx {camera_matrix[0, 2]} cy {camera_matrix[1, 2]}")
    logger.info(f"dist {dist}")
    plot(range(len(reproject_errors)), reproject_errors, title="Reprojection errors",
         ylabel="reprojection error", xlabel="image index")
    logger.info(f"avg error {np.mean(reproject_errors)} std {np.std(reproject_errors)}")

    return camera_matrix, dist


def rectify_image(image, camera_matrix, dist):
    """
    Undistort the image
    """
    image_height, image_width = image.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (image_width, image_height),
                                                         1, (image_width, image_height))
    image = cv2.undistort(image, camera_matrix, dist, None, new_camera_matrix)
    return image


def interactive(dataset_name: str):
    obj_pts, img_pts, image_sizes = find_correspondences((8, 6), 0.071,
                                                         load_files(get_dataset_by_name(
                                                             f"viewpro_calibration/{dataset_name}")),
                                                         True)
    camera_matrix, dist, reproject_error = calibrate(image_sizes, obj_pts, img_pts)
    obj_pts, img_pts, _ = filter_bad_calibration_images(obj_pts, img_pts, reproject_error, 5)
    camera_matrix, dist, reproject_error = calibrate(image_sizes, obj_pts, img_pts)

    for image in tqdm(load_files(get_dataset_by_name(f"viewpro_calibration/{dataset_name}"))):
        cv2.imshow('Preview', image)
        h, w = image.shape[:2]
        s = (w, h)
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, s, 1, s)
        cv2.imshow('Rectified', cv2.undistort(image, camera_matrix, dist, None, new_camera_matrix))
        cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    comparison_parser = subparsers.add_parser("comparison",
                                              help="Compare results of different sets of calibration")
    full_calibration_parser = subparsers.add_parser("fullcalibration",
                                                    help="Do the full calibration.")
    interactive_parser = subparsers.add_parser("interactive",
                                               help="Compute the full calibration matrix "
                                                    "and show the result on the chosen dataset.")
    interactive_parser.add_argument("dataset_name", type=str)

    args = parser.parse_args()
    if args.mode == "comparison":
        calibration_comparison([f"focus1dist1_5-3_m", "focus1dist2_5-3_m"])
        calibration_comparison([f"focus2dist1_6-3", "focus2dist2_6-3"])
        for f in [1, 2, 3]:
            calibration_comparison([f"focus{f}dist1_5-3", f"focus{f}dist2_5-3"])
        calibration_comparison([f"focus{f}dist1_5-3" for f in [1, 2, 3]])
        calibration_comparison([f"focus{f}dist2_5-3" for f in [1, 2, 3]])
    elif args.mode == "fullcalibration":
        full_calibration()
    elif args.mode == "interactive":
        # k, d = full_calibration()
        k = np.array([[1360, 0, 630], [0, 1360, 344], [0, 0, 1]])
        d = np.array([[-1.60867089e-01], [3.00941722e-01],
                      [5.51651036e-05], [6.23006522e-03],
                      [0.00000000e+00]])
        for image in load_files(get_dataset_by_name(f"viewpro_calibration/{args.dataset_name}")):
            cv2.imshow('Preview', image)
            cv2.imshow('Rectified', rectify_image(image, k, d))
            cv2.waitKey(100)
