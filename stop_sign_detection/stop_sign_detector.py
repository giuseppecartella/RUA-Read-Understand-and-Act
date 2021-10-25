import cv2
import numpy as np

class StopDetector():
    def __init__(self) -> None:
        pass

    def detect_stop_signal(self, template, rgb_img, KEYPOINTS_THRESHOLD=10):
        """
            Returns the approximate center of the stop signal, if found.
        """

        kp1, kp2, des1, des2 = self.compute_sift_imgs(template, rgb_img)        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        
        # Apply ratio test
        good , good_m = self.sift_ratio(matches)
        
        if len(good) < KEYPOINTS_THRESHOLD:
            return None

        keypoints_coords = [list(kp2[good_m[i].trainIdx].pt) for i in range(len(good))]
        keypoints_coords = np.array(keypoints_coords)
        return self.find_centre(keypoints_coords)

    def compute_sift_imgs(self, img1 , img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        return kp1, kp2, des1, des2

    def sift_ratio(self, matches):
        good = []
        good_m = []
        
        for m,n in matches:
            if m.distance < 0.72*n.distance:
                good.append([m])
                good_m.append(m)
        return good, good_m

    def find_centre(self, coords):
        """
        Inputs:
            coords: (N,2) matrix. N=number of keypoints.
                    Columns refer to x,y coords.
        """
        x_median = np.median(coords[:,0])
        y_median = np.median(coords[:,1])

        return round(x_median), round(y_median)

    def compute_distance(self, depth_img, center, FULL_WINDOW_SIZE):
        depth_img = depth_img.astype('float32')
        SIZE = int(FULL_WINDOW_SIZE/2)

        mask = np.where(depth_img==0, 255, 0).astype('uint8')
        depth_img = cv2.inpaint(depth_img, mask, 3, cv2.INPAINT_TELEA)

        x_c, y_c = center[0], center[1]
        depth_img = depth_img[y_c-SIZE:y_c+SIZE, x_c-SIZE:x_c+SIZE]

        # application median filter and removing borders
        #IL MEDIAN FILTER VA APPLICATO A TUTTA IMG E POI ESTRAGGO FINESTRA
        #O APPLICO SOLO ALL'INTERNO DELLA FINESTRA?
        #ATTUALMENTE LO APPLICHIAMO SOLO NELLA FINESTRA
        depth_img = cv2.medianBlur(depth_img, 3)

        removed_border = depth_img[1:-1, 1:-1]
        final_depth = cv2.copyMakeBorder(removed_border, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        
        return np.median(final_depth)