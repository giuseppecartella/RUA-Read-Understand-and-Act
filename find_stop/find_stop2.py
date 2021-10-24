import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import statistics
import random as rng
import math


def compute_sift_imgs(img1 , img2):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    return kp1, kp2, des1, des2

def find_centre_coords(coords):
    x = []
    y = []
    for c in coords:
        x.append(c[0])
        y.append(c[1])
    
    x_median = statistics.median(x)
    y_median = statistics.median(y)
    # the coordinates of a median point inside the signal
    # the one to pass as input to the robot 
    return x_median, y_median

def find_bbox_given_centre(img, x_centre, y_centre):
    # trovare idea migliore per prendere finestra attorno al punto centrale

    if x_centre > 60:
        h_x = 60
    else:
        h_x = round(x_centre)
    
    if y_centre > 60:
        h_y = 60
    else:
        h_y = round(y_centre) 
    
   
    masked_img = np.zeros_like(img)
    masked_img[round(y_centre)-h_y:round(y_centre)+h_y, round(x_centre)-h_x:round(x_centre)+h_x] = img[round(y_centre)-h_y:round(y_centre)+h_y, round(x_centre)-h_x:round(x_centre)+h_x]
    cv2.imshow('Masked Img', masked_img) 
    cv2.waitKey(0)
    
    signal_bbox_img = img[round(y_centre)-h_y:round(y_centre)+h_y, round(x_centre)-h_x:round(x_centre)+h_x]
    
    return  signal_bbox_img, masked_img, h_x, h_y

#Function to determine type of polygon on basis of number of sides
# POTREMMO CANCELLARE LE ALTRE SHAPE
def detectShape(cnt):          
    shape = 'unknown' 
    peri=cv2.arcLength(cnt,True) 
    vertices = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
    sides = len(vertices) 
    if (sides == 3): 
        shape='triangle' 
    elif(sides==4): 
            x,y,w,h=cv2.boundingRect(cnt)
            aspectratio=float(w)/h 
            if (aspectratio==1):
                shape='square'
            else:
                shape="rectangle" 
    elif(sides==5):
        shape='pentagon' 
    elif(sides==6):
        shape='hexagon' 
    elif(sides==8): 
        shape='octagon' 
    elif(sides==10): 
        shape='star'
    else:
        shape='circle' 
    return shape 

def find_box_max_area(src_gray , threshold=50):
    canny_output = cv2.Canny(src_gray, threshold, threshold * 3)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # the coordinates are returned in this way: (x, y , w , h)
    # (x,y) is the upper left point
    # w is width, h is height
    # for the area we just need base*height --> w* h
    print("coordinate dei box")
    print(boundRect)

    area = []
    for b in range(len(boundRect)):
        # Area
        area.append(boundRect[b][2] * boundRect[b][3])

    print("area ")
    print(area)

    # the biggest area is taken

    max = np.argmax(area)

    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv2.drawContours(drawing, contours_poly, max, color)
    cv2.rectangle(drawing, (int(boundRect[max][0]), int(boundRect[max][1])), \
                 (int(boundRect[max][0] + boundRect[max][2]), int(boundRect[max][1] + boundRect[max][3])), color, 5)

    cv2.imshow('Contours', drawing)
    cv2.waitKey()
  
    x = boundRect[max][0]
    y = boundRect[max][1]
    width = boundRect[max][2]
    height = boundRect[max][3]

def sift_ratio(matches):
    good = []
    good_m = []
    
    for m,n in matches:
    
        if m.distance < 0.72*n.distance:
            good.append([m])
            good_m.append(m)
    return good, good_m


def find_contour_max_area(contours):
    max_area = 0
    area = 0
    for cnt in contours:            
        shape=detectShape(cnt)
        
        if shape ==  'hexagon':            
            area = cv2.contourArea(cnt)
            if area >= max_area:
                max_area = area
                max_hex_cnt = [cnt]

    # da controllare che trovi almeno un esagono
    # se non lo trova ??? --> forse no probl perche trova comunuqe alcuni punti
    return max_hex_cnt

if __name__ == '__main__':
    template = cv2.imread('find_stop/template.jpg',0)
   
    # loading folders with images
    folder_rgb = 'find_stop/not'
    folder_depth = 'find_stop/depth'

    for rgb_img, depth_img in zip(os.listdir(folder_rgb),os.listdir(folder_depth)):
        
        img = cv2.imread(os.path.join(folder_rgb,rgb_img), 0)
 
        # Computing sift and keypoints
        kp1, kp2, des1, des2 = compute_sift_imgs(template, img)        

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        
        # Apply ratio test
        good , good_m = sift_ratio(matches)
        
        if len(good) > 10: # threshold number of keypoint found      

            coords = [kp2[good_m[i].trainIdx].pt for i in range(len(good))]
            
            x_centre , y_centre = find_centre_coords(coords)
        
            # draw lines for matching
            img3 = cv2.drawMatchesKnn(template,kp1,img,kp2,good,None,flags=2)

            cv2.imshow('Matches found',img3)
            cv2.waitKey(0)

            # ****************************metodo per trovare segnale
            
            # taking only a bbox around the centre both in a black img and in the real img
            signal_bbox_img, masked_img, h_x , h_y = find_bbox_given_centre(img, x_centre, y_centre) 
            
            # apllying canny in order to find countours
            edged = cv2.Canny(signal_bbox_img, 100,200)   
            plt.imshow(edged, cmap='gray')
            plt.show()                  

            (contours,_) = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

            # select only the countour with max area
            max_hex_cnt = find_contour_max_area(contours)  # ATTENZIONE CONTROLLA PROBLEMA !!!!         

            # drawing the polygon found           
            cv2.drawContours(signal_bbox_img, max_hex_cnt, -1,(0,255,0),2)  
            cv2.imshow('polygons_detected',signal_bbox_img)
            cv2.waitKey(0)

            # putting the found polygon in the coordinates of the not cropped img
            zero_mask = np.zeros_like(signal_bbox_img)
            cv2.drawContours(zero_mask, max_hex_cnt, -1, color=255, thickness=-1)

            masked_img[round(y_centre)-h_y:round(y_centre)+h_y, round(x_centre)-h_x:round(x_centre)+h_x] = zero_mask
            cv2.imshow('Final Result',masked_img)
            cv2.waitKey(0)       
            
            # ***********************depth part       
            # remember: depth_img and img have the same shape (480,640) 
            #applicare hole filling + median filter (see Bigazzi and Landi 's paper)
            
            depth_array = np.load(os.path.join(folder_depth,depth_img))

            signal_pts = np.where(masked_img == 255)
            signal_depth = depth_array[signal_pts]
            
            #print(statistics.mean(signal_depth))
            #print( len(np.where( signal_depth == 0)))
            
            coord_img = (x_centre, y_centre, statistics.mean(signal_depth) )
            print(coord_img)
        else:
            
            print("Stop signal not found!")