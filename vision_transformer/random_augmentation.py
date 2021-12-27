import numpy as np
import cv2
import random

#RANDOM OPERATIONS:
# 1) sign upscale and downscale in a range. First, define default sign size
# 2) rotate sign of a random angle deleting black background from the result
# 3) change sign brightness randomly
# 4) apply noise to the background with a certain probability (i.e 50%)
# 5) random homografy to change perspective
# 6) sign flip
# 7) background flip


class RandomImgOperator():
    def scale_sign(self, background, sign):
        scale_factor = round(random.uniform(0.5, 2), 2)
        sign = cv2.resize(sign, None, fx=scale_factor, fy=scale_factor)
        return background, sign

    def rotate_sign(self, background, sign):
        image_center = tuple(np.array(sign.shape[1::-1]) / 2)
        angle = random.randint(-70, 70)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        sign = cv2.warpAffine(sign, rot_mat, sign.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
        sign = cv2.resize(sign, (200,100))
        return background, sign

    def change_brightness(self, background, sign):
        alpha = round(random.uniform(0.5, 2.5), 2)
        if sign is not None:
            sign = cv2.convertScaleAbs(sign, alpha=alpha, beta=0)
        background = cv2.convertScaleAbs(background, alpha=alpha, beta=0)
        return background, sign


    def apply_noise(self, background, sign):
        background = cv2.GaussianBlur(background, (11,11), sigmaX=10, sigmaY=10)
        return background, sign

    def random_homography(self, background, sign):
        pts1 = np.float32([[0, 0], [200,0], [200,100], [0, 200]])
        option_a = np.float32([[0, 0], [200,0], [200,100], [0, 350]])
        option_b = np.float32([[0, -50], [200,0], [200, 200], [0, 200]])
        pts2 = random.choice([option_a, option_b])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(sign, matrix, (200, 100), borderValue=(255,255,255))
        return background, result


    def flip_background(self, background, sign):
        background = cv2.flip(background, random.choice([0,1]))
        return background, sign

    def merge(self, background, img):
        if img is None:
            return background
        
        #merge the sign on a random position inside the background
        h_s, w_s, _ = img.shape
        h_b, w_b, _ = background.shape
        start_width = random.randint(0, w_b - w_s - 1)
        start_heigth = random.randint(0, h_b - h_s - 1)

        background[start_heigth:start_heigth+h_s, start_width:start_width+w_s] = img
        return background


    def apply_random_operations(self, background=None, img=None, only_background=False):
        if only_background:
            options = [0, self.flip_background, self.apply_noise, self.change_brightness]
            results = np.random.choice(options, 1)
        else: 
            funcs = [self.scale_sign, self.rotate_sign, self.change_brightness, 
                        self.apply_noise, self.random_homography, self.flip_background]

            results = np.random.choice(funcs, 2, replace=False)

            if results[-1] == self.random_homography:
                results = results[::-1]
            
        for res in results:
            if res == 0:
                break
            else:
                background, img = res(background, img)

        result = self.merge(background, img)
        return result