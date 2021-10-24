import numpy as np
import cv2
import os

def fillPoints(imgDepth):
    pad = 15
    imgDepth = cv2.copyMakeBorder(imgDepth, pad, pad, pad, pad, 0)

    for i in range(pad,imgDepth.shape[0]-pad):
        for j in range(pad,imgDepth.shape[1]-pad):
            # Let we suppose to take 15x15 window
            if imgDepth[i,j] == 0:
                neighbors = imgDepth[i-pad:i+pad+1,j-pad:j+pad+1].flatten()
                neighbors = np.delete(neighbors, np.where(neighbors == 0))
                imgDepth[i,j] = np.median(neighbors)

    return imgDepth[pad:imgDepth.shape[0]-pad, pad:imgDepth.shape[1]-pad]

if __name__ == '__main__':
    # loading folders with images
    folder_rgb = 'rgb'
    folder_depth = 'depth'

    for rgb_img, depth_img in zip(os.listdir(folder_rgb), os.listdir(folder_depth)):
        imgDepth = np.load(os.path.join(folder_depth, depth_img), 0)
        #imgRGB = cv2.imread(os.path.join(folder_rgb, rgb_img), 0)

        out = fillPoints(imgDepth)
        np.save(f'Holes_filled/{imgDepth}', out)
