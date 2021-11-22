import numpy as np
import cv2
from . import project_parameters as params

class SignalDetector():
    def __init__(self, window_size=16, lab_mode="False"):
        self.lab_mode = lab_mode
        self.WINDOW_SIZE = window_size
        self.template = cv2.cvtColor(cv2.imread('utils/template.jpg'), cv2.COLOR_BGR2RGB)

    def _compute_sift_imgs(self, img1, img2):
        if self.lab_mode == "True":
            sift = cv2.xfeatures2d.SIFT_create()
        else:
            sift = cv2.SIFT_create()
        
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        return kp1, kp2, des1, des2

    def _sift_ratio(self, matches):
        good = []
        good_m = []

        for m, n in matches:
            if m.distance < 0.72 * n.distance:
                good.append([m])
                good_m.append(m)
        return good, good_m

    def _find_centre_coords(self, coords):
        coords = np.array(coords)
        x_median = np.median(coords[:,0])
        y_median = np.median(coords[:,1])
        
        return int(round(x_median)), int(round(y_median))

    def look_for_signal(self, rgb_image):
        print("Computing SIFT on the image...")
        kp1, kp2, des1, des2 = self._compute_sift_imgs(self.template, rgb_image)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good, good_m = self._sift_ratio(matches)

        if len(good) < params.SIFT_MATCH_THRESHOLD:
            return False, None, None
        else:
            coords = [kp2[good_m[i].trainIdx].pt for i in range(len(good))]
            x_c, y_c = self._find_centre_coords(coords)
            return True, x_c, y_c

    def get_signal_distance(self, x_c, y_c, depth_image):
        HALF_WINDOW_SIZE = int(self.WINDOW_SIZE / 2)

        bottom_limit = y_c - HALF_WINDOW_SIZE
        upper_limit = y_c + HALF_WINDOW_SIZE
        left_limit = x_c - HALF_WINDOW_SIZE
        right_limit = x_c + HALF_WINDOW_SIZE
        final_depth = depth_image[bottom_limit:upper_limit, left_limit:right_limit]
        depth_value = np.median(final_depth)

        return depth_value