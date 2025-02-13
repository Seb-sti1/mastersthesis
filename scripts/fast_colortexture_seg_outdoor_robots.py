"""
It based on https://ieeexplore.ieee.org/document/4651086

This was written with the help of https://github.com/tbjszhu/.
"""
import logging
import os

import cv2
import numpy as np
import ot
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from tqdm import tqdm

from scripts import resize_images, load_files, show_image, show_images

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "aukerman")
logger = logging.getLogger(__name__)


def compute_descriptor(image: np.ndarray, neighbor_size: int) -> np.ndarray:
    """
    See III equation (1) of Fast Color/Texture Segmentation For Outdoor Robots
    :param image: The image (in LAB color) to compute descriptor for
    :param neighbor_size: The size of the neighborhood around the pixel
    :return: the descriptor of each pixel
    """
    h, w, _ = image.shape
    k = (neighbor_size - 1) // 2
    w1, w2, w3 = 0.5, 1.0, 0.5

    descriptor = np.zeros((h - 2 * k, w - 2 * k, 11), dtype=np.float32)

    for i in range(k, h - k):
        for j in range(k, w - k):
            descriptor[i - k, j - k, 0:3] = image[i, j, 0:3] * [w1, w2, w3]
            neighbor = w3 * (np.absolute(image[i, j, 0] - image[i - k:i + k + 1, j - k:j + k + 1, 0])
                             .reshape(neighbor_size ** 2))
            descriptor[i - k, j - k, 3:(neighbor_size ** 2 - 1) // 2 + 3] = neighbor[0:(neighbor_size ** 2 - 1) // 2]
            descriptor[i - k, j - k, (neighbor_size ** 2 - 1) // 2 + 3:] = neighbor[(neighbor_size ** 2 - 1) // 2 + 1:]

    return descriptor


def compute_histograms(clusters_desc: np.ndarray, n_clusters_desc: int, histogram_window: int) -> np.ndarray:
    h, w = clusters_desc.shape

    integral_images = np.zeros((h + 1, w + 1, n_clusters_desc), dtype=np.uint8)
    for i in range(n_clusters_desc):
        integral_images[:, :, i] = cv2.integral((clusters_desc == i).astype(np.uint8))

    h, w, n_clusters = integral_images.shape
    hist = np.zeros((h - histogram_window, w - histogram_window, n_clusters), dtype=np.uint8)

    for i in range(histogram_window // 2, h - histogram_window // 2):
        for j in range(histogram_window // 2, w - histogram_window // 2):
            i_local, j_local = i - histogram_window // 2, j - histogram_window // 2
            # top-left
            hist[i_local, j_local, :] += integral_images[i - histogram_window // 2, j - histogram_window // 2, :]

            # top-right
            hist[i_local, j_local, :] -= integral_images[i - histogram_window // 2, j + histogram_window // 2, :]

            # bottom-right
            hist[i_local, j_local, :] += integral_images[i + histogram_window // 2, j + histogram_window // 2, :]

            # bottom-left
            hist[i_local, j_local, :] -= integral_images[i + histogram_window // 2, j - histogram_window // 2, :]

    if logger.level == logging.DEBUG:
        show_images([integral_images[:, :, i] / integral_images[:, :, i].max() for i in range(n_clusters_desc)]
                    + [hist[:, :, i] for i in range(n_clusters_desc)],
                    (2, n_clusters_desc),
                    f"Integral and histogram images for each clusters")
    return hist


def main(n_clusters_desc: int, n_clusters_hist: int,
         neighbor_size: int, histogram_window: int,
         emd_threshold: float, outlier_threshold: float):
    # colors to show clusters
    cmap = plt.get_cmap("tab20", n_clusters_desc)
    colors = (cmap(np.arange(max(n_clusters_desc, n_clusters_hist)))[:, :3] * 255).astype(np.uint8)
    edge = neighbor_size // 2

    for img in tqdm(resize_images(load_files(DATASET_PATH,
                                             lambda p: p.endswith(".JPG")),
                                  max_width=1000,
                                  max_height=1000)):
        h, w, _ = img.shape

        # === compute color/texture descriptors
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_descriptor = compute_descriptor(img_lab, neighbor_size)
        img_descriptor_list = img_descriptor.reshape(((h - 2 * edge) * (w - 2 * edge), 3 + neighbor_size ** 2 - 1))

        # === k-means on descriptor
        # TODO See if KMeans can be optimized
        #   > In the K-means iterations, reclassification attempts are not made
        #   > for points that lie less than half the mean absolute distance
        #   > away from their currently classified center
        kmeans_desc = KMeans(n_clusters_desc, random_state=4465132)
        kmeans_desc.fit(img_descriptor_list)
        textons = kmeans_desc.cluster_centers_
        clusters_desc = kmeans_desc.predict(img_descriptor_list).reshape(((h - 2 * edge), (w - 2 * edge)))

        # > Once the 16 textons for a given image have been established,
        # > each pixel is classified as belonging to one of
        # > these using Euclidean distance. A simple threshold identifies outliers
        outliers = np.linalg.norm(img_descriptor - textons[clusters_desc], axis=2)
        clusters_desc[outliers > outlier_threshold] = -1

        if logger.level == logging.DEBUG:
            plt.hist(outliers.reshape((h - 2 * edge) * (w - 2 * edge)), bins=20, edgecolor="black")
            plt.xlabel("Distance from centroid")
            plt.ylabel("Frequency")
            plt.title("Histogram of outliers")
            plt.axvline(x=outlier_threshold, color='r', linestyle='--', linewidth=2, label="Outlier threshold")
            plt.legend()
            plt.show()

        # === histogram clustering
        histogram = compute_histograms(clusters_desc, n_clusters_desc, histogram_window)
        histogram_list = histogram.reshape((histogram.shape[0] * histogram.shape[1], n_clusters_desc))
        kmeans_hist = KMeans(n_clusters_hist, random_state=4465132)
        kmeans_hist.fit(histogram_list)
        clusters_hist = kmeans_hist.predict(histogram_list).reshape(histogram.shape[:2])

        # === merge clusters with too small earth mover's distance
        clusters = clusters_hist.copy()
        distance_matrix = cdist(textons, textons)
        clusters_mapping = {i: i for i in range(n_clusters_hist)}
        clusters_emd_distance = np.zeros((n_clusters_hist, n_clusters_hist))  # only the superior triangle is filled
        for i in range(n_clusters_hist):
            for j in range(i + 1, n_clusters_hist):
                clusters_emd_distance[i, j] = ot.emd2(
                    kmeans_hist.cluster_centers_[i] / kmeans_hist.cluster_centers_[i].sum(),
                    kmeans_hist.cluster_centers_[j] / kmeans_hist.cluster_centers_[j].sum(),
                    distance_matrix)
                if clusters_emd_distance[i, j] < emd_threshold:
                    clusters_mapping[j] = clusters_mapping[i]

        if logger.level == logging.DEBUG:
            show_image(clusters_emd_distance, "Histogram cluster EMD distances")
            logger.debug(f"{clusters_emd_distance}")

        for i in range(n_clusters_hist):
            if clusters_mapping[i] != i:
                clusters[clusters == i] = clusters_mapping[i]

        if logger.level == logging.DEBUG:
            show_image(img, "Raw Image")
            show_image(img_lab, "LAB Image")

            legend = {i: {'color': colors[i], 'name': str(i)} for i in range(n_clusters_desc)}
            legend.update({-1: {'color': colors[-1], 'name': 'outliers'}})
            show_image(colors[clusters_desc], "Clusters (before histogram)", legend)

            legend = {i: {'color': colors[i], 'name': str(i)} for i in clusters_mapping.keys()}
            show_image(colors[clusters_hist], "Clusters (before emd)", legend)
            show_image(colors[clusters], "Clusters (final)", legend)

        # === display
        alpha = 0.4
        overlay = np.zeros_like(img)
        overlay[histogram_window // 2:-histogram_window // 2 - 1,
        histogram_window // 2:-histogram_window // 2 - 1, :] = colors[clusters]
        cv2.imshow("Clusters", cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0))
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)
    # paper param 16, 8, 3, 6, 100, ?
    hist_window = 16
    assert hist_window % 2 == 0, "Histogram window must be even"
    neighbor_size = 3
    assert neighbor_size % 2 == 1, "Neighbor size must be odd"

    main(4, 6,
         neighbor_size, hist_window,
         8, 175)
