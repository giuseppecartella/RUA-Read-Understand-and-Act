import numpy as np
import cv2

#da spiegare a sara e nicolas:
#get_signal_distance() is the old compute_depth_distance()
#non serve più l'inpaint perchè lo facciamo subito quando 
#faccio get_rgb_depth()

class SignalDetector():
    def __init__(self, template, window_size=16):
        self.WINDOW_SIZE = window_size
        self.template = template

    def _compute_sift_imgs(self, img1, img2):
        sift = cv2.xfeatures2d.SIFT_create() #needed for Python 2
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

        if len(good) < 10:
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