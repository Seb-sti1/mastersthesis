from pathlib import Path

import cv2

from scripts.utils.datasets import load_paths_and_files, get_dataset_by_name, resize_image

seq1 = Path("norlab_ulaval_datasets/test_dataset/sequence1/")


def synchronized_iterators():
    ugv_iter = load_paths_and_files(get_dataset_by_name(seq1 / "ground/images"))
    ugv_reproject_iter = load_paths_and_files(get_dataset_by_name(seq1 / "ground/projections"))
    uav_iter = load_paths_and_files(get_dataset_by_name(seq1 / "aerial/images"))

    curr_ugv = ugv_iter.__next__()
    curr_ugv_time = int(curr_ugv[0].stem)
    next_ugv = ugv_iter.__next__()
    next_ugv_time = int(next_ugv[0].stem)

    curr_ugv_reprojected = ugv_reproject_iter.__next__()

    curr_uav = uav_iter.__next__()
    curr_uav_time = int(curr_uav[0].stem)

    start_time = min(int(curr_ugv[0].stem), int(curr_uav[0].stem))

    try:
        while True:
            yield ((curr_ugv_time - start_time, curr_ugv[1]),
                   curr_ugv_reprojected[1],
                   (curr_uav_time - curr_ugv_time, curr_uav[1]))

            if abs(curr_uav_time - curr_ugv_time) < abs(curr_uav_time - next_ugv_time):
                curr_uav = uav_iter.__next__()
                curr_uav_time = int(curr_uav[0].stem)
            curr_ugv = next_ugv
            next_ugv = ugv_iter.__next__()
            curr_ugv_reprojected = ugv_reproject_iter.__next__()
            curr_ugv_time = int(curr_ugv[0].stem)
            next_ugv_time = int(next_ugv[0].stem)
    except StopIteration:
        pass


def show_uav_ugv():
    is_running = True

    for ((ugv_time, ugv), ugv_reproject, (uav_time, uav)) in synchronized_iterators():
        img = resize_image(ugv, 600)
        cv2.putText(img, f"{ugv_time / 1_000_000_000}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"ugv", img)
        cv2.imshow("ugv_reproject", resize_image(ugv_reproject, 600))

        img = resize_image(resize_image(uav, 600), 600)
        cv2.putText(img, f"{uav_time / 1_000_000_000}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"uav", img)
        k = cv2.waitKey(1 if is_running else 0) & 0xFF
        if k == ord('q'):
            break
        elif k == ord(' '):
            is_running = not is_running


if __name__ == '__main__':
    show_uav_ugv()
