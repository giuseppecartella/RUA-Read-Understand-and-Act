import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()

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

def main():
    template = cv2.imread('find_stop/template.jpg', 0)

    # loading folders with images
    folder_rgb = 'find_stop/rgb'
    folder_depth = 'find_stop/depth'

    idx = 0
    found_sign = 0
    for rgb_img, depth_img in zip(os.listdir(folder_rgb),os.listdir(folder_depth)):
        print(f'Analyzing image number {idx}')
        idx += 1

        img = cv2.cvtColor(cv2.imread(os.path.join(folder_rgb,rgb_img)), cv2.COLOR_BGR2RGB)

        kp1, kp2, des1, des2 = compute_sift_imgs(template, img)        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        
        # Apply ratio test
        good , good_m = sift_ratio(matches)
        
        if len(good) < 10:
            print('Stop signal not found!')
            continue #analyze next img
        
        found_sign += 1
        coords = [kp2[good_m[i].trainIdx].pt for i in range(len(good))]
        x_c, y_c = find_centre_coords(coords)
        #plt.imshow(img)
        #plt.scatter([x_c],[y_c], c='blue', marker='+', s=100)
        #plt.show()
        
        """
        WINDOW_SIZE = 16
        HALF_WINDOW_SIZE = int(WINDOW_SIZE/2)

        x_c, y_c = round(x_c), round(y_c)
        tl = (x_c-HALF_WINDOW_SIZE, y_c+HALF_WINDOW_SIZE) #top left corner
        br = (x_c+HALF_WINDOW_SIZE, y_c-HALF_WINDOW_SIZE)
        img = cv2.rectangle(img, tl, br, (255,0,0), 2)
        plt.imshow(img)
        plt.show()
        """

        #----------distance computation----------#
        WINDOW_SIZE = 16
        HALF_WINDOW_SIZE = int(WINDOW_SIZE/2)

        x_c, y_c = round(x_c), round(y_c)
        depth = np.load(os.path.join('find_stop', 'depth', depth_img)).astype('float32')
        plot_img(depth)

        #INPAINTING
        #define masks. Points where depth is = 0.
        mask = np.where(depth==0, 255, 0).astype('uint8')
        plot_img(mask)

        dst = cv2.inpaint(depth, mask, 3, cv2.INPAINT_TELEA)
        plot_img(dst)

        depth = depth[y_c-HALF_WINDOW_SIZE:y_c+HALF_WINDOW_SIZE, x_c-HALF_WINDOW_SIZE:x_c+HALF_WINDOW_SIZE]
        plot_img(depth)
        
        # application median filter and removing borders
        median = cv2.medianBlur(depth, 3)
        plot_img(median)
        
        removed_border = median[1:-1, 1:-1]
        final_depth = cv2.copyMakeBorder(removed_border, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        plot_img(final_depth)
        
        #now first of all we should detect outliers which values are far from the mean
        #then manage possible holes and as last step compute mean distance from the stop sign.
        ...
    
    print(f'Found {found_sign} signs!')

if __name__ == '__main__':
    main()