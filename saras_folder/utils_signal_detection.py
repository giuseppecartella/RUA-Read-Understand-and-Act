import numpy as np
import cv2
import os

class Detection_Helper():

    def compute_sift_imgs(img1 , img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        return kp1, kp2, des1, des2

    def sift_ratio(matches):
        good = []
        good_m = []
        
        for m,n in matches:
            if m.distance < 0.72*n.distance:
                good.append([m])
                good_m.append(m)
        return good, good_m

    def find_centre_coords(coords):
        x = []
        y = []
        for c in coords:
            x.append(c[0])
            y.append(c[1])
        
        x_median = np.median(x)
        y_median = np.median(y)

        return x_median, y_median

    def look_for_signal(self, rgb_image, template):
        kp1, kp2, des1, des2 = self.compute_sift_imgs(template, rgb_image)        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        
        # Apply ratio test
        good , good_m = self.sift_ratio(matches)
        
        if len(good) < 10:
            #print('Stop signal not found!')
            
            return False, None, None
        else:
            coords = [kp2[good_m[i].trainIdx].pt for i in range(len(good))]
            x_c, y_c = self.find_centre_coords(coords)
            return True, x_c, y_c

    def compute_depth_distance(x_c, y_c):
        WINDOW_SIZE = 16
        HALF_WINDOW_SIZE = int(WINDOW_SIZE/2)

        x_c, y_c = round(x_c), round(y_c)
        depth = np.load(os.path.join('find_stop', 'depth', depth_img)).astype('float32')

        #INPAINTING
        #define masks. Points where depth is = 0.
        mask = np.where(depth==0, 255, 0).astype('uint8')    
        dst = cv2.inpaint(depth, mask, 3, cv2.INPAINT_TELEA)

        depth = dst[y_c-HALF_WINDOW_SIZE:y_c+HALF_WINDOW_SIZE, x_c-HALF_WINDOW_SIZE:x_c+HALF_WINDOW_SIZE]
        
        # application median filter and removing borders
        median = cv2.medianBlur(depth, 3)
        
        removed_border = median[1:-1, 1:-1]
        final_depth = cv2.copyMakeBorder(removed_border, 1, 1, 1, 1, cv2.BORDER_REFLECT)       

        depth_value = np.median(final_depth)
        return depth_value
        
