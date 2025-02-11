"""

https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#getting-started
"""

import os

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from scripts.u_net_segmentation_test import load_images, resize_images

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "aukerman")


class DetectClick:

    def __init__(self, img: np.ndarray):
        self.coord = None
        self.img = img

    def run(self):
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)
        while not self.coord:
            cv2.imshow("Image", self.img)
            cv2.waitKey(1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coord = (x, y)


if __name__ == "__main__":
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join(os.path.dirname(__file__), "sam_vit_b_01ec64.pth"))
    predictor = SamPredictor(sam)

    for img in tqdm(resize_images(load_images(DATASET_PATH))):
        valid = False

        while not valid:
            c = DetectClick(img)
            c.run()

            try:
                predictor.set_image(img)
                masks, _, _ = predictor.predict(point_coords=np.array([c.coord]), point_labels=np.array([1]))
            except Exception as e:
                print(e)
                continue

            overlay = np.zeros_like(img)
            for i in range(masks.shape[0]):
                mask = (masks[i] * 255).astype(np.uint8)
                color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
                overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            alpha = 0.5
            blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

            cv2.imshow("Masks Overlay", overlay)
            cv2.imshow("Image", blended)

            k = cv2.waitKey(0) & 0xFF
            if k == ord('v'):
                valid = True
            elif k == ord('q'):
                break
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()
